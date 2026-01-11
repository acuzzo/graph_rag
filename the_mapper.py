

"""
 
-
"""
from __future__ import annotations
import os
from dotenv import load_dotenv; load_dotenv()
import re
from datetime import datetime, date
from typing import Optional, Dict, Any, List
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
from openai import OpenAI
import faiss
import numpy as np
import hashlib
import fsspec
from pathlib import Path
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct
import unicodedata
from uuid import uuid5, NAMESPACE_URL
from sklearn.metrics import classification_report, accuracy_score
from rapidfuzz import fuzz
import unicodedata
from transformers import pipeline
import torch
import json



# ----------------------------
# Core classes
# ---------------------------- 
class TheMapper():
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_uri = config["io"]["base_uri"]
        self.vector_id_column = self.config["vector_store"]["id_column"]
        self.master_uri = f"{self.base_uri}{self.config["paths"]["master"]["path"]}"
        self.input_uri = f"{self.base_uri}{self.config["paths"]["input"]["path"]}"

        # object composition
        self.transform = Transform(self.input_uri, self.config)
        
        self.mapping_file_manager = MappingFileManager(
            self.transform.df_affiliations_long, 
            config
        )
        self.semantic_matcher = SemanticMatcher(
            self.master_uri,
            self.mapping_file_manager.mapping_unified,
            self.config,
        )           
     
        # delegation
        self.df_affiliations_long = self.transform.df_affiliations_long
        self.mapping_unified = self.mapping_file_manager.mapping_unified
                
    def _load_master_validated(self) -> pd.DataFrame:
        """Load master and enforce duplicate constraints (independent of SemanticMatcher)."""
        df_master = pd.read_parquet(self.master_uri)
        df_master[self.vector_id_column] = df_master[self.vector_id_column].astype("string")
        dupes = pd.concat([
            df_master[df_master.duplicated(subset=["name", "country", "city", "state_region"], keep=False)],
            df_master[df_master.duplicated(subset=["match_field"], keep=False)],
            df_master[df_master.duplicated(subset=[self.vector_id_column], keep=False)],
        ])
        
        if dupes.shape[0] != 0:
            raise AssertionError("dupes found in master list")
        
        return df_master

    def apply_fuzzy_match(self, build: bool = False):
        # don't rebuild the collection every time as we are doing dump and re-load atm. 
        # TODO: Move to incremental load in the long run
        if build:
            self.semantic_matcher.build()

        df_matched_unvalidated = self.semantic_matcher.apply_match(
            self.mapping_file_manager.mapping_unified
        )

        if df_matched_unvalidated is None or df_matched_unvalidated.shape[0] == 0:
            # print ("no rows to review, skips semantic match, applied final mapping")
            # return self.apply_final_mapping()
            print("no rows to match go to apply_final_mapping")
            return None
        else:
            self.mapping_file_manager.append_and_export_mapping(df_matched_unvalidated)

    def apply_final_mapping(self):
        df_mapping = self.mapping_file_manager.get_df_mapping()
        # df_mapping = self.mapping_file_manager.mapping_unified
        df_master  = self._load_master_validated()
        return self.transform.apply_final_mapping(
            df_mapping,
            df_master
        )

class SemanticMatcher():
    def __init__(self, master_uri: str, mapping_unified: pd.DataFrame, config: Optional[Dict[str, Any]] = None):
        self.config = (config or {})
        self.open_ai = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) # official OpenAI Python client
        self.qdrant = QdrantClient(host="qdrant", port=6333, prefer_grpc=True, timeout=30)
        # self.qdrant = QdrantClient(url="http://localhost:6333", timeout=30)
                
        self.master_uri = master_uri
        self.vector_threshold = float(self.config["vector_store"]["threshold"])
        self.vector_column = self.config["vector_store"]["vector_column"]
        self.vector_payload_columns = self.config["vector_store"]["payload_columns"]
        self.vector_id_column = self.config["vector_store"]["id_column"]
        self.qdrant_collection_name = self.config["vector_store"]["collection_name"]
        self.df_master = self.get_df_master()


    def _search(self, query, k):
        # vectorise the search term
        response = self.open_ai.embeddings.create(model="text-embedding-3-large", input=query)
        search_vectors = response.data[0].embedding
        
        search_result = self.qdrant.query_points(
            collection_name=self.qdrant_collection_name,
            query=search_vectors,
            limit=k,
            with_payload=True,
            with_vectors=False,
            # score_threshold=k
            search_params=models.SearchParams(hnsw_ef=256, exact=False) 
            )
        
        return_values = []
        for p in search_result.points: 
            # TODO!!! add self.vector_column "match_field" to payload then no need to lookup master !!!!!!
            search_result_string = self.df_master.loc[
                self.df_master[self.vector_id_column] == p.id, self.vector_column
            ].iloc[0]
            search_result_string
      
            return_values.append({
                self.vector_id_column: p.id, 
                "text": search_result_string, 
                "score": float(p.score), 
                "payload": p.payload
            })
        
        # return out
        return return_values

    def build(self):
        df_master_list_additions = (self.df_master[self.df_master[self.vector_column].notna()])
        df_master_list_additions = df_master_list_additions.where(pd.notna(df_master_list_additions), None)
        df_master_list_additions["payload"] = (
            df_master_list_additions[self.vector_payload_columns]
            .to_dict("records")
        )

        # recreate collection for full refresh (long run want to incremental load and not do this)
        if self.qdrant.collection_exists(self.qdrant_collection_name):
            self.qdrant.delete_collection(self.qdrant_collection_name)

        self.qdrant.create_collection(
            self.qdrant_collection_name, 
            vectors_config=VectorParams(
                size=3072, #text-embedding-3-large
                distance=Distance.COSINE, 
                on_disk=True
            ),
            hnsw_config=models.HnswConfigDiff(m=32, ef_construct=256)
        )
            
        # vectorise in batches (openai fails over a certain threshold)
        y = 500
        for x in tqdm(range(0, len(df_master_list_additions), y), desc="building vectors"):
            # Step 1: Get embeddings from OpenAI
            df_master_list_additions_batch = df_master_list_additions.iloc[x: x + y]
            texts = df_master_list_additions_batch[self.vector_column].tolist()
            ids = df_master_list_additions_batch[self.vector_id_column].tolist()
            payload = df_master_list_additions_batch["payload"].tolist()
            
            response = self.open_ai.embeddings.create(model="text-embedding-3-large", input=texts)
            upsert_points = [
                models.PointStruct(id=v[0], vector=v[1], payload=v[2])
                for v 
                in list(zip(
                    ids, 
                    [d.embedding for d in response.data],
                    payload
                ))
            ]
                            
            upsert_operation = models.UpsertOperation(upsert=models.PointsList(points=upsert_points))
            
            self.qdrant.batch_update_points(
                collection_name=self.qdrant_collection_name,
                update_operations=[upsert_operation],
                wait=True
            )
            
        # sanity check: count points
        cnt = self.qdrant.count(self.qdrant_collection_name, exact=True).count
        print("Indexed points:", cnt)

    def _rf_rerank(self, query: str, cands: list[dict]) -> list[dict]:
        """Rerank top-k candidates using RapidFuzz lexical score + tiny payload boosts."""
        # !!!TODO use df not dicts speed things up
        q = (query or "").casefold()
        for c in cands:
            t = (c.get("text") or "").casefold()
            # normalize to 0..1
            lex = fuzz.token_set_ratio(q, t) / 100.0

            payload = c.get("payload") or {}
            country = (payload.get("country") or "").casefold()
            city    = (payload.get("city") or "").casefold()

            boost = 0.0
            # Small nudges if query literally contains these tokens
            if country and country in q: boost += 0.05
            if city and city in q:       boost += 0.03

            # Dense score is primary; lexical + boosts are tie-breakers
            c["final"] = float(c["score"]) + 0.20 * lex + boost

        return sorted(cands, key=lambda x: x["final"], reverse=True)

    def get_df_master(self) -> pd.DataFrame:
                
        df_master = pd.read_parquet(self.master_uri)
        df_master[self.vector_id_column] = df_master[self.vector_id_column].astype("string")
        # Validate duplicates
        dupes = pd.concat([
            df_master[df_master.duplicated(subset=["name", "country", "city", "state_region"], keep=False)],
            df_master[df_master.duplicated(subset=["match_field"], keep=False)],
            df_master[df_master.duplicated(subset=[self.vector_id_column], keep=False)]
        ])
        if dupes.shape[0] != 0:
            raise AssertionError("dupes found in master list")
        return df_master
    
    def lookup_master(self, _row: pd.Series) -> pd.Series:
        query = _row["affiliation_clean"]

        # Guard against NA / None / empty strings BEFORE calling _search
        if query is None or pd.isna(query) or str(query).strip() == "":
            _row["match_field"] = None
            _row["match_score"] = None
            _row["date_added"]  = None
            return _row
        
        # Normal path: only run if query is valid
        cands = self._search(query, k=10) 
        
        if not cands:
            _row["match_field"] = None
            _row["match_score"] = None
            _row["date_added"]  = None
            return _row

        # Rerank with RapidFuzz
        cands = self._rf_rerank(query, cands)
        top = cands[0]  

        _row["match_field"] = top["text"]
        _row["match_score"] = float(top["score"])
        _row["date_added"]  = datetime.now().strftime("%Y-%m-%d %H:%M")
        # _row["confidence"] = "high" if _row["match_score"] >= 0.85 else ("med" if _row["match_score"] >= 0.78 else "low")

        return _row

    def apply_match(self, mapping_unified: pd.DataFrame) -> pd.DataFrame:
  
        # Work only on unvalidated rows 
        unvalidated = mapping_unified[
            mapping_unified["validated"].fillna("n").astype(str).str.lower() == "n"
        ]
        
        if unvalidated.shape[0] == 0:
            return unvalidated
        else:
        
            list_rows = [_row for _, _row in unvalidated.iterrows()]

            max_workers = int(self.config.get("concurrency", 30))
            rows = list(
                tqdm(
                    ThreadPoolExecutor(max_workers=max_workers).map(self.lookup_master, list_rows),
                    total=len(list_rows),
                    desc="applying semantic match"
                )
            )
            
            master_columns = [ self.vector_id_column, 'match_field', "name", "country", 'link_ROR_v2_1','id_ror']
            self.df_matched_unvalidated = (
                pd.DataFrame(rows)
                .merge(
                    # TODO on id_uuid instead of match field?
                    self.df_master[self.df_master["match_field"].notna()].filter(master_columns), 
                    on="match_field", 
                    how="left"
                )
                .filter([self.vector_id_column, "affiliation_clean", "validated", 'match_field', 
                        "name", "country", "match_score", 'date_added', 'link_ROR_v2_1', 'id_ror'])
                .rename(columns={
                    self.vector_id_column: "id_mapping",
                    "name": "matched_name",
                    "country": "matched_country"
                    }
                )
            )
        
            self.df_matched_unvalidated = self.df_matched_unvalidated.drop_duplicates(
                subset=["affiliation_clean"], keep="first"
            )
      

        return self.df_matched_unvalidated

class MappingFileManager:
    def __init__(self, df_affiliations_long: pd.DataFrame, config: Dict[str, Any]):
        self.config = config
        self.base_uri = config["io"]["base_uri"]
        self.mapping_uri = f"{self.base_uri}{self.config["paths"]["mapping"]["path"]}"
        ts = datetime.now().strftime('%Y%m%d-%H%M')
        self.backup_uri = config["io"]["backup_uri"]
        self.mapping_filename = self.config["paths"]["mapping"]["path"].split("/")[-1].split(".")[0]
        self.mapping_backup_uri = f"{self.backup_uri}/{self.mapping_filename}_{ts}.parquet.gzip"
        self.mapping_to_val = f"{self.base_uri}{self.config["paths"]["to_val"]["path"]}"
        self.df_affiliations_long = df_affiliations_long
        
        self.df_mapping = self.get_df_mapping()
        self.df_mapping_additions = self.get_df_mapping_additions()
        self.mapping_unified = self.get_mapping_unified()

    def get_df_mapping(self) -> pd.DataFrame:
        print("get_df_mapping")
        # Load mapping if it exists; otherwise return empty schema to create new file
        try:
            return pd.read_parquet(self.mapping_uri)
        except FileNotFoundError:
            return pd.DataFrame(
                columns=[
                    'id_mapping',
                    'affiliation_clean',
                    'validated', 
                    'match_field',
                    'matched_name',
                    'matched_country',
                    'match_score',
                    'date_added',
                    "link_ROR_v2_1",
                    "id_ror"
                ]
            )
           
    def get_df_mapping_additions(self) -> pd.DataFrame:
        print("get_df_mapping_additions")
        
        # First pass: use full mapping on affiliation_clean 
        # (affiliation_clean = concat of name + all location fields from affil string)
        base = (
            self.df_affiliations_long
            .merge(
                self.df_mapping,
                on='affiliation_clean',
                how='left',
                suffixes=("", "_map"),
                validate='many_to_one'
            )
        )

        # instead of full affil string we try and match on ORG only or ORG + COUNTRY etc.
        # to improve mapping hit rate 
        def _try_fill_with_key(base_df, left_key):
            """
            For rows with id_mapping == NaN, try to fill from self.df_mapping
            using df_affiliations_long[left_key] -> self.df_mapping['affiliation_clean'].
            Only updates rows that:
            - are currently unmapped
            - find a non-null id_mapping in self.df_mapping
            """
            mask_missing = base_df['id_mapping'].isna()
            if not mask_missing.any():
                return base_df

            subset = base_df.loc[mask_missing]

            extra = subset.merge(
                self.df_mapping,
                left_on=left_key,
                right_on='affiliation_clean',
                how='left',
                suffixes=("_base", ""),
                validate='many_to_one'
            )

            fill_mask = extra['id_mapping'].notna()
            if not fill_mask.any():
                return base_df

            idx = subset.index[fill_mask]

    
            base_df.loc[idx, 'id_mapping'] = extra.loc[fill_mask, 'id_mapping'].values
           
            return base_df
        
        # fallback matching priority after initial affiliation_clean match:
        # 2) org_country
        # 3) ORG
        base = _try_fill_with_key(base, left_key='org_country')
        base = _try_fill_with_key(base, left_key='ORG')

        # Anything STILL missing id_mapping goes to semantic search / mapping additions
        df_mapping_additions = (
            base.loc[base["id_mapping"].isna()]
            .filter([
                'id_mapping', 'affiliation_clean', 'validated',
                'match_field',
                'matched_name', 'matched_country',
                'match_score', 'date_added', 'link_ROR_v2_1', 'id_ror',
            ])
            .drop_duplicates(keep='first')
            .assign(validated=lambda _df: _df["validated"].fillna("n"))
        )
        
        # ----------  rows that GOT an id_mapping via fallback matching - auto variants ----------
        # 
        auto_variants = (
            base.loc[base["id_mapping"].notna(), ["affiliation_clean", "id_mapping"]]
            .drop_duplicates()
        )

        # keep only variants that are not already in the mapping table
        auto_variants = auto_variants[
            ~auto_variants["affiliation_clean"].isin(self.df_mapping["affiliation_clean"])
        ]

        # join back to self.df_mapping on id_mapping to pull metadata
        auto_variants = auto_variants.merge(
            self.df_mapping.drop(columns=["affiliation_clean"]),
            on="id_mapping",
            how="left"
        )

        # mark them as validated 
        auto_variants["validated"] = auto_variants["validated"].fillna("y")

        # stash for later use
        self.df_auto_variants = auto_variants

        return df_mapping_additions
     # find affiliations not present in mapping
        # base = (
        #     self.df_affiliations_long
        #     .merge(self.df_mapping, on='affiliation_clean', how='left', suffixes=("", "_map"))
        # )
        
        # df_mapping_additions = (
        #     base.loc[base["id_mapping"].isna()]
        #     .filter(['id_mapping','affiliation_clean','validated',
        #             'match_field','matched_name','matched_country',
        #             'match_score','date_added','link_ROR_v2_1', 
        #             'id_ror', 
        #             ])
        #     .drop_duplicates(keep='first')
        #     .assign(validated=lambda _df: _df["validated"].fillna("n"))
        # )
        
        # return df_mapping_additions

    def get_mapping_unified(self) -> pd.DataFrame:
        print("get_mapping_unified")
        # list of dfs 
        frames = [self.df_mapping]

        # add auto variants if available
        if hasattr(self, "df_auto_variants") and self.df_auto_variants is not None:
            if not self.df_auto_variants.empty:
                frames.append(self.df_auto_variants)

        # add new unmapped rows for semantic search
        frames.append(self.df_mapping_additions)

        mapping_unified = (
            pd.concat(frames, ignore_index=True)
            .assign(validated=lambda _df: _df["validated"].fillna("n"))
            .assign(id_mapping=lambda _df: _df["id_mapping"].astype("string"))
            .drop_duplicates(subset=["affiliation_clean"], keep='first')
        )
        
        # mapping_unified = (
        #     pd.concat([self.df_mapping_additions, self.df_mapping], ignore_index=True)
        #     .assign(validated=lambda _df: _df["validated"].fillna("n"))
        #     .assign(id_mapping=lambda _df: _df["id_mapping"].astype("string"))
        #     .drop_duplicates(subset=["affiliation_clean"], keep='first')
        # )
    
        return mapping_unified
    
    def commit_fallback_matches(self):
        print("commit_fallback_matches")
        """
        Persist current validated mappings (including fallback/ORG/org_country
        auto-variants) to the parquet mapping file, without adding new
        semantic/embedding-based matches.
        """
        # Existing keys before this commit
        existing_affils = set(self.df_mapping["affiliation_clean"].astype("string"))

        # All validated rows (old + new)
        keep_validated = self.mapping_unified[self.mapping_unified["validated"] != "n"].copy()

        # Identify which rows are NEW (affiliation_clean not in previous mapping)
        mask_new = ~keep_validated["affiliation_clean"].astype("string").isin(existing_affils)

        # Only set date_added for new rows
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        keep_validated.loc[mask_new, "date_added"] = now_str

        keep_validated["date_added"] = (
            pd.to_datetime(keep_validated["date_added"], errors="coerce")
            .dt.strftime("%Y-%m-%d %H:%M")
        )

        dupes = keep_validated[keep_validated.duplicated(subset=["affiliation_clean"], keep=False)]
        if not dupes.empty:
            raise AssertionError("dupes found in mapping in commit_fallback_matches")

        # Backup old parquet, then overwrite
        self.df_mapping.to_parquet(self.mapping_backup_uri, compression="gzip")
        keep_validated.to_parquet(self.mapping_uri, compression="gzip")
        print(f"mapping file (fallback-only commit) exported: {self.mapping_uri}")

# !!!! not good takes too long on 78 mins still going why???....
        # Refresh in-memory state
        self.df_mapping = keep_validated.copy()
        self.df_mapping_additions = self.get_df_mapping_additions()
        self.mapping_unified = self.get_mapping_unified()


    def append_and_export_mapping(self, df_matched_unvalidated: pd.DataFrame):
        print("append_and_export_mapping")
        # Merge new matches with validated rows
        keep_validated = self.mapping_unified[self.mapping_unified["validated"] != "n"]
        df_export = pd.concat([df_matched_unvalidated, keep_validated], ignore_index=True)
        # df_export["date_added"] = (
        #     pd.to_datetime(df_export["date_added"], errors="coerce")
        #     .dt.strftime("%Y-%m-%d %H:%M")
        # )
        
        # only new additions new date added
        n_new = len(df_matched_unvalidated)
        mask_new = df_export.index < n_new

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        df_export.loc[mask_new, "date_added"] = now_str

        df_export["date_added"] = (
            pd.to_datetime(df_export["date_added"], errors="coerce")
            .dt.strftime("%Y-%m-%d %H:%M")
        )

        dupes = df_export[df_export.duplicated(subset=["affiliation_clean"], keep=False)]
        if dupes.shape[0] != 0:
            raise AssertionError("dupes found in mapping")

        
        # backup then export mapping for manual review
        self.df_mapping.to_parquet(self.mapping_backup_uri, compression="gzip")
        df_export.to_parquet(self.mapping_uri, compression="gzip")
        print(f"mapping file exported for review: {self.mapping_uri}")
        
        # ---- append new unvalidated rows to existing Excel does not overwrite ----
        try:
            df_existing_to_val = pd.read_excel(self.mapping_to_val)
        except FileNotFoundError:
            df_existing_to_val = pd.DataFrame(columns=df_matched_unvalidated.columns)

        df_to_val = (
            pd.concat([df_existing_to_val, df_matched_unvalidated], ignore_index=True)
            .drop_duplicates(subset=["affiliation_clean"], keep="first")
        )

        df_to_val.to_excel(self.mapping_to_val, index=False)
        print(f"unvalidated mapping file exported for review/validating: {self.mapping_to_val}")

        # keep in-memory state in sync
        self.df_mapping = df_export.copy()
        self.df_mapping_additions = self.get_df_mapping_additions()
        self.mapping_unified = self.get_mapping_unified()
    
    def merge_validated_to_parquet_mapping(self):
        print("merge_validated_to_parquet_mapping")
        df_mapping = pd.read_parquet(self.mapping_uri)
        df_mapping_to_val = pd.read_excel(self.mapping_to_val)
        df_mapping_validated = df_mapping_to_val[df_mapping_to_val['validated'] == 'y'].copy()
        
        # affiliation_clean as the key
        df_mapping = df_mapping.set_index("affiliation_clean")
        df_mapping_validated = df_mapping_validated.set_index("affiliation_clean")

        # Only update overlapping columns
        common_cols = df_mapping.columns.intersection(df_mapping_validated.columns)
        df_mapping.update(df_mapping_validated[common_cols])

        # Restore index as a column
        df_mapping = df_mapping.reset_index()

        # Write updated mapping back to parquet
        df_mapping.to_parquet(self.mapping_uri, index=False)
        print(f"Updated mapping parquet written to: {self.mapping_uri}")
        
        # --- Remove validated rows from df_mapping_to_val and write back to Excel ---
        df_remaining = df_mapping_to_val[df_mapping_to_val["validated"] != "y"].copy()

        df_remaining.to_excel(self.mapping_to_val, index=False)
        print(f"Remaining unvalidated mapping rows written back to: {self.mapping_to_val}")
    
class Transform:
    def __init__(self, input_uri: str, config: dict):
        print("Transform:init")
        
        self.config = config
        self.input_uri = input_uri
        self.source = config["source"]
        
        # ---------- NER CONFIG ----------
        ner_cfg = config.get("NER", {})
        device_cfg = ner_cfg.get("device", "auto")
        
        # map config device 
        if device_cfg == "cpu":
            self.ner_device = -1
            self.ner_batch_size = ner_cfg.get("batch_size_cpu", 8)

        elif device_cfg == "cuda" or str(device_cfg).startswith("cuda"):
            # "cuda" or "cuda:1" etc
            if ":" in str(device_cfg):
                gpu_idx = int(str(device_cfg).split(":")[1])
            else:
                gpu_idx = 0
            self.ner_device = gpu_idx
            self.ner_batch_size = ner_cfg.get("batch_size_cuda", 32)

        elif device_cfg == "auto":
            if torch.cuda.is_available():
                self.ner_device = 0      # first GPU
                self.ner_batch_size = ner_cfg.get("batch_size_cuda", 32)
            else:
                self.ner_device = -1     # CPU
                self.ner_batch_size = ner_cfg.get("batch_size_cpu", 8)

        elif isinstance(device_cfg, int):
            # allow direct gpu index in config: "device": 1
            self.ner_device = device_cfg
            self.ner_batch_size = ner_cfg.get("batch_size_cuda", 32)

        else:
            raise ValueError(f"Unknown NER device config: {device_cfg}")
        
        
        suffix = Path(input_uri).suffix.lower()  
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(input_uri)
        elif suffix in (".json"):
            df= pd.read_json(input_uri)
        elif suffix in (".csv"):
            df= pd.read_csv(input_uri)
        
        self.df_input_data = df.rename(columns=lambda col: str(col).strip())
        # self.df_input_data = pd.read_excel(input_uri).rename(columns=lambda col: str(col).strip())
        # self.df_input_data = pd.read_json(input_uri)
        
        self.country_col = "COUNTRY"
        self.vector_id_column = self.config["vector_store"]["id_column"]
        self.country_master = config["paths"]["country_master"]["path"]
        self.country_aliases = config["paths"]["country_aliases"]["path"]
        self.df_country_master = pd.read_json(self.country_master)
        self.df_country_aliases = pd.read_json(self.country_aliases, typ="series")
        self.sub_phrases_url = config["paths"]["sub_phrases"]["path"]
        
        # dispatch to source transform
        self.df_affiliations_long = getattr(self, f"transform_{self.source}", None)()
    
    def clean_affil_general(self, df: pd.DataFrame, column: str, out_col: str = "affiliations_clean_block"):
        print("Transform:clean_affil_general")
        """
            Clean an affiliation-like text column and write it to `out_col`.
            - Normalizes unicode (NFKC)
            - Trims brackets/quotes
            - Normalizes newlines
            - Removes footnote marks/superscripts
            - Collapses spaces and strips around newlines
        """
        df[out_col] = (
            df[column]
                .astype("string")
                .str.strip()
                .str.replace(r'^\[|\]$', '', regex=True)
                .str.strip(' "\'')
                .str.replace(r'\\r\\n|\\n', '\n', regex=True)
                .str.replace(r'\r', '\n', regex=True)
                .str.normalize("NFKC")
                .str.replace(r'[\*\u2020\u2021\u00A7\u00B6\u00B2\u00B3\u00B9\u2070-\u2079\u207A-\u207F]+', '', regex=True)
                .str.replace(r'[ \t]+', ' ', regex=True)
                .str.replace(r'[ \t]*(\n)[ \t]*', r'\1', regex=True)
        )
        return df 
    
    def explode_first_step(self, df_cleaned, sep_pat, auth_pat):
        print("Transform:explode_first_step")
        """
        sep_pat pattern splits auth/afilliaton
        auth_pat pattern splits auth from affiliation
        rest could include department, email, social media etc
        
        output varies depending on source it could be list of authors 
        or list of affiliations
        
        returns df with two new columns author and rest
         
        """
        df_exp = df_cleaned.assign(
        blocks=(
            df_cleaned['affiliations_clean_block']
            .astype('string')
            .str.strip()
            .str.split(sep_pat, regex=True)  
        )
        ).explode('blocks', ignore_index=True)
        
        # clean the exploded text 
        df_exp['blocks'] = (
            df_exp['blocks'].astype('string').str.strip(" '\"\t ")
        )
        
        df_exp[['author','rest']] = (
        df_exp['blocks'].astype('string')
            .str.split(auth_pat, n=1, expand=True, regex=True)
        )
        return df_exp
        
    def ner_affiliation(self, df_exp, list_columns, identifier_columns, col_apply_ner):
        print("Transform:ner_affiliation")
        """
        NER on affiliation text, returning ONE ORG + ONE COUNTRY per row.
        - One output row per ORG span (per affiliation block).
        - COUNTRY (and CITY/REGION/SUB) are attached to the nearest ORG by position.
        - Identifier columns (e.g. surrogate_id, author) are preserved.
        - affil_id is treated as a stable affiliation-level key /affil_id (FK to df_exp).
        """


            # require affil_id to be present 
        if "affil_id" not in df_exp.columns:
            raise ValueError("df_exp must contain 'affil_id' before calling ner_affiliation")

        df_ner_id_text = df_exp[list_columns].copy()

        ner = pipeline(
            task="ner",
            model="SIRIS-Lab/affilgood-NER",
            aggregation_strategy="simple",
            device=self.ner_device,
        )
        out = ner(df_ner_id_text[col_apply_ner].astype(str).tolist(), batch_size=self.ner_batch_size)

        # Sanity check: same number of results as input rows
        assert len(out) == len(df_ner_id_text), "NER output length does not match input rows."
        keep = ["SUB", "ORG", "CITY", "REGION", "COUNTRY"]

        # Flatten ner spans, index by affil_id
        spans_s = pd.Series(
            out,
            index=df_ner_id_text["affil_id"],
            name="spans"
        ).explode().dropna()

        spans_df = pd.json_normalize(spans_s)
        spans_df["affil_id"] = spans_s.index

        spans_df = spans_df.loc[spans_df["entity_group"].isin(keep)].copy()
        spans_df["word"] = spans_df["word"].astype(str).str.strip(" ,;")

        # ---- relabel known ORG→SUB mislabels  ----
        with open(self.sub_phrases_url) as f:
            org_to_sub_list = json.load(f)
        org_to_sub = {s.strip().lower() for s in org_to_sub_list if s.strip()}
        spans_df["word_norm"] = spans_df["word"].str.lower().str.strip()
        mask_org_to_sub = (
            (spans_df["entity_group"] == "ORG")
            & spans_df["word_norm"].isin(org_to_sub)
        )
        spans_df.loc[mask_org_to_sub, "entity_group"] = "SUB"
        spans_df = spans_df.drop(columns=["word_norm"], errors="ignore")

        # Attach identifiers (surrogate_id, author, auth_position, ...)
        spans_df = spans_df.merge(
            df_ner_id_text[["affil_id"] + identifier_columns],
            on="affil_id",
            how="left"
        )

        # set affil_id index
        id_lookup = (
            df_ner_id_text[["affil_id"] + identifier_columns]
            .drop_duplicates("affil_id")
            .set_index("affil_id")
        )

        org_rows = []

        # For each affiliation block (affil_id), build one row per ORG
        for affil_id, grp in spans_df.groupby("affil_id", sort=False):
            base_ids = id_lookup.loc[affil_id].to_dict()

            org_spans     = grp[grp["entity_group"] == "ORG"].sort_values("start")
            sub_spans     = grp[grp["entity_group"] == "SUB"].sort_values("start")
            city_spans    = grp[grp["entity_group"] == "CITY"].sort_values("start")
            region_spans  = grp[grp["entity_group"] == "REGION"].sort_values("start")
            country_spans = grp[grp["entity_group"] == "COUNTRY"].sort_values("start")

            # --- SUB handling strategy ---

            # True when pattern is like SUB, SUB, ORG, ORG:
            # all SUB spans occur before the first ORG span
            all_subs_before_all_orgs = (
                (not sub_spans.empty)
                and (not org_spans.empty)
                and (sub_spans["start"].max() < org_spans["start"].min())
            )

            # For the "no ORG at all" fallback we still want block-level SUB agg
            if not sub_spans.empty:
                sub_agg_block = " | ".join(
                    sub_spans["word"].dropna().drop_duplicates().tolist()
                )
            else:
                sub_agg_block = pd.NA

            # helper to find nearest span of a given type
            def _nearest_word(span_df, pos):
                if span_df.empty:
                    return pd.NA
                idx = (span_df["start"] - pos).abs().argmin()
                return span_df.iloc[idx]["word"]

            # If no ORG at all, still emit one row so we don't lose the block
            if org_spans.empty:
                row = {**base_ids}
                row.update({
                    "affil_id": affil_id,
                    "SUB": sub_agg_block,
                    "ORG": pd.NA,
                    "CITY": _nearest_word(city_spans, 0),
                    "REGION": _nearest_word(region_spans, 0),
                    "COUNTRY": _nearest_word(country_spans, 0),
                })
                org_rows.append(row)
                continue

            # For the first ORG, use -inf so it can capture all SUBs before it 
            # (not just after position 0)
            prev_org_start = float("-inf")

            for _, org in org_spans.iterrows():
                org_start = org["start"]

                # ----- SUB per ORG -----
                if sub_spans.empty:
                    sub_for_org = pd.NA
                elif all_subs_before_all_orgs:
                    # e.g. SUB, SUB, ORG, ORG -> all ORGs share all SUBs
                    sub_for_org = sub_agg_block
                else:
                    # Take SUBs that appear after the previous ORG and up to this ORG
                    mask_sub = (
                        (sub_spans["start"] <= org_start)
                        & (sub_spans["start"] > prev_org_start)
                    )
                    subs_slice = sub_spans.loc[mask_sub, "word"].dropna().drop_duplicates()
                    if subs_slice.empty:
                        # fallback: use nearest preceding SUB, or leave as NA
                        sub_for_org = pd.NA
                    else:
                        sub_for_org = " | ".join(subs_slice.tolist())

                row = {**base_ids}
                row.update({
                    "affil_id": affil_id,
                    "SUB": sub_for_org,
                    "ORG": org["word"],
                    "CITY": _nearest_word(city_spans, org_start),
                    "REGION": _nearest_word(region_spans, org_start),
                    "COUNTRY": _nearest_word(country_spans, org_start),
                })
                org_rows.append(row)
                prev_org_start = org_start

        df_ner = pd.DataFrame(org_rows)

        # Ensure all expected columns are present (even if empty)
        for c in ["SUB", "ORG", "CITY", "REGION", "COUNTRY"]:
            if c not in df_ner.columns:
                df_ner[c] = pd.NA

           
        return df_ner

    def norm(self, s: pd.Series):
        print("Transform:norm")
        # Unicode normalize + strip accents
        s = s.astype(str).map(lambda x: unicodedata.normalize("NFKD", x))
        s = s.map(lambda x: x.encode("ascii", "ignore").decode("ascii"))
        # lowercase + keep letters/digits as space-separated tokens
        s = s.str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()
        return s
    
    def map_alpha2_country_code(self, df_ner):
        print("Transform:map_alpha2_country_code")
        # slim-2.json should be an array of objects (records)
        slim_df = self.df_country_master[["name", "alpha-2"]].copy()
        slim_df["key"] = self.norm(slim_df["name"])

        # (index = normalized name, value = 2 letter code (alpha-2))
        base_map = slim_df.set_index("key")["alpha-2"]

        # alias dict as a Series, normalize keys
        alias_s = self.df_country_aliases.copy()
        alias_s.index = self.norm(alias_s.index.to_series())
        alias_s = alias_s.astype("string").str.upper()

        combined = pd.concat([base_map, alias_s])
        combined = combined[~combined.index.duplicated(keep="last")]
        # map
        df_ner["alpha2_country_code"] = self.norm(df_ner[self.country_col]).map(combined)
        
        df_ner_alpha2 = df_ner.copy()
        
        missing_codes = (
        df_ner_alpha2.loc[
            df_ner_alpha2[self.country_col].notna() & df_ner_alpha2['alpha2_country_code'].isna(),
            self.country_col
        ]
        .drop_duplicates()
        .tolist()
        )

        return df_ner_alpha2, missing_codes

    def country_name_map(self, path, df_ner_alpha2):
        print("Transform:country_name_map")
        # Map code -> country name
        code_to_name = (
            self.df_country_master.drop_duplicates("alpha-2")
                        .set_index("alpha-2")["name"]
        )

        df_ner_alpha2["alpha2_country_name"] = (
            df_ner_alpha2["alpha2_country_code"].astype("string").str.upper().map(code_to_name)
        )

        df_ner_alpha2_name = df_ner_alpha2.copy()
        
        return df_ner_alpha2_name       


    
# ---------------------------
#            PUBMED
# ---------------------------

    def transform_pubmed(self) -> pd.DataFrame:
        print("Transform:transform_pubmed")
    
        def add_pmid_num(df, col):
            s = df[col].astype('string').str.strip()

            token_pat = r'(?i)(?:pmid\s*:\s*)?\d{6,}'
            counts = s.str.count(token_pat)

            first_num = s.str.extract(r'(?i)(?:pmid\s*:\s*)?(\d{6,})')[0]

            df['PMID_num'] = (
                pd.to_numeric(first_num.where(counts == 1), errors='coerce')
                .astype('Int64')
            )
            return df

        def add_doi_url(df, col):
            s = df[col].astype('string').str.strip()

            doi_core_pat = r'(?i)\b(10\.\d{4,9}/[^\s"<>]*[A-Za-z0-9])'
            counts = s.str.count(doi_core_pat)

            doi_core = s.str.extract(doi_core_pat, expand=False).str.lower()
            doi_core = doi_core.where(counts == 1).astype('string')

            df['doi_url'] = 'https://doi.org/' + doi_core
            return df

        EMAIL_PAT = r'(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
        NAME_CORE = r'[A-ZÀ-ÖØ-Ý][\w\'\.\-\sÀ-ÖØ-öø-ÿ]{1,80}'
        sep_pat =  r'\s*\|\|\|\s*'
        auth_pat = r'\s*--\s*'
        
        #  NER perameters 
        identifier_columns = ["surrogate_id", "author", "auth_position"]
        col_apply_ner = "affil"
        # TODO some refactoring here as redudency !!!!!
        list_columns = [
            "affil_id",
            "surrogate_id",
            "author",
            "auth_position",
            "affil",
        ]
        
        # convert to proper timestamp
        # self.df_input_data['Publication Date'] = pd.to_datetime(
        #     self.df_input_data['Publication Date'], unit='ms', utc=True
        # )
        
        self.df_input_data["Publication Date"] = (
            pd.to_datetime(self.df_input_data["Publication Date"], unit="ms", utc=True)
            .dt.tz_localize(None)
        )

        
        self.df_input_data['year'] = self.df_input_data['Publication Date'].dt.year
        
        # 1-based integer IDs (article-level)
        self.df_input_data["surrogate_id"] = np.arange(
            1, len(self.df_input_data) + 1, dtype=np.int64
        )
        
        # helper function clean up pmid / doi
        add_pmid_num(self.df_input_data, 'PMID')
        add_doi_url(self.df_input_data, 'DOI')

        df_cleaned = self.clean_affil_general(self.df_input_data, 'Authors_Affiliations')
        df_cleaned["affiliations_clean_block"] = (
            df_cleaned['affiliations_clean_block']
            .str.replace(
                rf'\.\s*,\s+(?=(?:["\']?){NAME_CORE}\s--\s)',
                '. ||| ',
                regex=True
            )
            .str.replace(
                rf',\s+(?=(?:["\']?){NAME_CORE}\s--\s)',
                ' ||| ',
                regex=True
            )
        )
        
        self.df_exp = self.explode_first_step(df_cleaned, sep_pat, auth_pat)
        
        self.df_exp['author'] = self.df_exp['author'].str.strip(' .;,"\'')
        self.df_exp['auth_position'] = (
            self.df_exp.groupby('surrogate_id').cumcount() + 1
        )

        # explode affiliation pieces
        self.df_exp = self.df_exp.assign(
            affil=self.df_exp['rest']
                .str.split(r'\s*(?:;\s*|\.\s*\n+|\n{2,})\s*')
        ).explode('affil')

        # **********************************
        # extract email and clean it out of affil
        # **********************************
        EMAIL_PAT_NOFLAGS = r'\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b'
        self.df_exp['email'] = (
            self.df_exp['affil']
            .str.extract(rf'({EMAIL_PAT_NOFLAGS})', flags=re.I, expand=False)
            .str.lower()
        )
        self.df_exp['affil'] = (
            self.df_exp['affil']
            .str.replace(EMAIL_PAT_NOFLAGS, '', flags=re.I, regex=True)
            .str.strip(' .;,\'"')
        )

        # drop entirely empty affiliations 
        self.df_exp = self.df_exp[
            self.df_exp['affil'].notna()
            & (self.df_exp['affil'].astype(str).str.strip() != "")
        ]

        # **********************************
        # affiliation-level indices
        # **********************************
        
        # affil_id: global stable ID per affiliation row
        self.df_exp['affil_id'] = np.arange(
            len(self.df_exp), dtype='int64'
        ) 

     
        # **********************************
        # NER on affiliation
        # **********************************
        df_ner = self.ner_affiliation(
            self.df_exp,
            list_columns=list_columns,
            identifier_columns=identifier_columns,
            col_apply_ner=col_apply_ner,
        )

        # country mapping / harmonisation
        df_ner_country_code, self.missing_codes = self.map_alpha2_country_code(df_ner)
        df_harmonised_country = self.country_name_map(self, df_ner_country_code)
        df_harmonised_country = self.country_name_map(self.country_master, df_harmonised_country)
        
        # concat: 'affiliation_clean' used for mapping to ROR
        s_org  = df_harmonised_country["ORG"].astype("string").str.strip()
        s_city = df_harmonised_country["CITY"].astype("string").str.strip()
        s_reg  = df_harmonised_country["REGION"].astype("string").str.strip()
        s_ctry = (
            df_harmonised_country["alpha2_country_name"]
            .fillna(df_harmonised_country["COUNTRY"])
            .astype("string")
            .str.strip()
        )

        def add_part(base, part):
            base = base.fillna("")
            part = part.fillna("")
            return base + np.where(
                (base != "") & (part != ""),
                ", " + part,
                part
            )

        aff = add_part(add_part(add_part(s_org, s_city), s_reg), s_ctry)
        df_harmonised_country["affiliation_clean"] = aff.mask(
            aff.str.strip().eq(""), pd.NA
        )

        # merge back article/affiliation-level info from df_exp
        cols_from_exp = [
            "affil_id",
            "email",
            "Publication Date",
            "year",
            "PMID_num",
            "doi_url",
        ]
        df_harmonised_country = df_harmonised_country.merge(
            self.df_exp[cols_from_exp],
            on="affil_id",
            how="left",
            validate="many_to_one"
        )

        # inst_idx = ORG index per (article, author), in the order they appear
        df_harmonised_country["inst_idx"] = (
            df_harmonised_country
            .groupby(["surrogate_id", "author", "auth_position"], sort=False)
            .cumcount() + 1
        )

        # add concat of ORG and COUNTRY for mapping
        df_harmonised_country['org_country'] = (
            df_harmonised_country['ORG'].astype("string") + ', ' +
            df_harmonised_country['COUNTRY'].astype("string")
        )

        # final column order
        cols = [
            "surrogate_id",
            "auth_position",
            "author",
            "affil_id",    # stable FK to df_exp
            "inst_idx",    # ORG index per author, in text order
            "SUB",
            "ORG",
            "CITY",
            "REGION",
            "COUNTRY",
            "alpha2_country_code",
            "alpha2_country_name",
            "org_country",
            "affiliation_clean",
            "email",
            "Publication Date",
            "year",
            "PMID_num",
            "doi_url",
        ]

        df_harmonised_country = df_harmonised_country.filter(cols)

        # natural keys:
        # - ORG level: (surrogate_id, auth_position, inst_idx) or (affil_id, inst_idx)
        return df_harmonised_country
 
# ---------- EMBASE ----------

    def transform_embase(self) -> pd.DataFrame:
        print("Transform:transform_embase")
        sep_pat = r'\.\s*,\s*(?=\()'  # period, comma, then next char is "("
        # split immediately AFTER the first closing bracket, eating any spaces
        auth_pat = r'(?<=[\)\]）])\s+' # supports ), ], and full-width ）
        list_columns = ["Abstract_ID",  "rest", "author"]
        identifier_columns = ["Abstract_ID", "author"]
        groupby_columns = ["Abstract_ID", "author"]
        col_apply_ner = "rest"
        
        df_cleaned = self.clean_affil_general(self.df_input_data, 'INSTITUTIONS')
        self.df_exp = self.explode_first_step(df_cleaned, sep_pat,  auth_pat)
        df_ner = self.ner_affiliation(self.df_exp, list_columns, identifier_columns, groupby_columns, col_apply_ner)

        df_ner_country_code, self.missing_codes = self.map_alpha2_country_code(df_ner)
        df_harmonised_country = self.country_name_map(self, df_ner_country_code)
        
        df_harmonised_country = self.country_name_map(self.country_master, df_harmonised_country)
        
        # concat: 'affiliation_clean' used for mapping to ROR-a bit weird as had to solve for NAs 
        s_org  = df_harmonised_country["ORG"].astype("string").str.strip()
        s_city = df_harmonised_country["CITY"].astype("string").str.strip()
        s_reg  = df_harmonised_country["REGION"].astype("string").str.strip()
        s_ctry = df_harmonised_country["alpha2_country_name"]\
                    .fillna(df_harmonised_country["COUNTRY"])\
                    .astype("string").str.strip()

        def add_part(base, part):
            base = base.fillna("")
            part = part.fillna("")
            return base + np.where((base != "") & (part != ""), ", " + part, part)

        aff = add_part(add_part(add_part(s_org, s_city), s_reg), s_ctry)
        df_harmonised_country["affiliation_clean"] = aff.mask(aff.str.strip().eq(""), pd.NA)
        
        return df_harmonised_country       
    

    
# ---------- Final mapping dispatcher ----------
    def apply_final_mapping(self, df_mapping: pd.DataFrame, df_master: pd.DataFrame) -> pd.DataFrame:
        print("Transform:apply_final_mapping")
        final_mapping_function = getattr(self, f"apply_final_mapping_{self.source}", None)
        return final_mapping_function(df_mapping, df_master)

    # ---------- EMBASE final mapping  ----------
    def apply_final_mapping_embase(self, df_mapping: pd.DataFrame, df_master: pd.DataFrame):
        print("Transform:apply_final_mapping_embase")
        #
        # Merge mapping and master, then explode authors
        df_mapped = (
            self.df_affiliations_long
            # .merge(df_mapping, on="affiliation_clean", how="left")
            .merge(df_mapping.loc[df_mapping['validated']== 'y'], on="affiliation_clean", how="left")
            .merge(df_master, left_on="id_mapping", right_on=self.vector_id_column, how="left", suffixes=("", "_master"))
        )

        # Select available columns safely
        desired = ["Id", "Parent_affiliation", "AUTHORS_extracted", "name", "country", "city", "state_region", "types"]
        cols = [c for c in desired if c in df_mapped.columns]
        df_mapped = df_mapped.loc[:, cols]

        # Explode authors
        if "AUTHORS_extracted" in df_mapped.columns:
            df_mapped = df_mapped.explode("AUTHORS_extracted", ignore_index=True)
            # Split tuple into columns
            if "AUTHORS_extracted" in df_mapped.columns:
                tmp = pd.DataFrame(df_mapped["AUTHORS_extracted"].tolist(), columns=["author_order", "author"])
                df_mapped = pd.concat([df_mapped.drop(columns=["AUTHORS_extracted"]), tmp], axis=1)

        # Final renaming
        if "Id" in df_mapped.columns:
            df_mapped = df_mapped.rename(columns={"Id": "abstract_id"})
        if "name" in df_mapped.columns:
            df_mapped = df_mapped.rename(columns={"name": "institution"})

        return df_mapped
        

    # ----------  final mapping pubmed  ----------
    def apply_final_mapping_pubmed(self, df_mapping: pd.DataFrame, df_master: pd.DataFrame) -> pd.DataFrame:
        print("Transform:apply_final_mapping_pubmed")
        
        # (first_name,second_name, title, footnote_marker, combind_name, ORCID, email,)
        # surrogate key + source for global key
        
   # Merge mapping and master,
        affil_long_cols = ['surrogate_id', 'auth_position', 'author', 'orcid','inst_idx', 
                              'SUB', 'ORG','CITY', 'REGION', 'COUNTRY', 'alpha2_country_code',
                              'alpha2_country_name', 'email'
                            # 'org_country', 
                            # 'affiliation_clean'
                            ]
    
        # place holder!
        self.df_affiliations_long['orcid'] = None
        
        df_mapped = (
            self.df_affiliations_long
            .merge(df_mapping, on="affiliation_clean", how="left")
            # .merge(df_mapping.loc[df_mapping['validated']== 'y'], on="affiliation_clean", how="left")
            # id_uuid
            .merge(df_master, left_on="id_mapping", right_on=self.vector_id_column, how="left", suffixes=("", "_master"))
            .filter(affil_long_cols + ["id_ror_master", 'name', 'country_code']
                    + ["match_score"]
                    )
            .rename(columns={
                "id_ror_master": "id_ror",
                'name': 'ror_name',
                'country_code': 'ror_country_code'
                             })
        )
        
        df_mapped["match_score"] = df_mapped["match_score"].round(2)
        
        # json ready  **********************************************
        
        
        # pd.set_option("display.max_colwidth", None)
        aff_cols = [
            'auth_position', 'author', 'orcid', 'inst_idx', 'SUB',
            'ORG', 'CITY', 'REGION', 'COUNTRY', 'alpha2_country_code',
            'alpha2_country_name', 'email', 'id_ror', 'ror_name',
            'ror_country_code', 'match_score'
        ]

        df_json_ready = df_mapped.copy()
        df_json_ready[aff_cols] = df_json_ready[aff_cols].replace({pd.NA: None, np.nan: None})

        row_dicts = df_json_ready[aff_cols].to_dict(orient="records")

        tmp = df_json_ready[['surrogate_id']].copy()
        tmp['auth_affiliation_harmonised'] = row_dicts

        nested = (
            tmp.sort_values('surrogate_id')
            .groupby('surrogate_id', sort=False)['auth_affiliation_harmonised']
            .agg(list)
            .reset_index()
        )
        # nested


        df_input_with_harm = self.df_input_data.merge(nested, on='surrogate_id', how='left', validate='one_to_one')
        

        df_input_with_harm.drop(
            columns=['year', 
                     'surrogate_id', 
                     'PMID_num', 'doi_url', 'affiliations_clean_block'],
            inplace=True
        )

        
        return df_input_with_harm

class MasterManager:
    """
    Processes raw ROR data into a cleaned, deduplicated master table 
    (with UUIDs and match fields) and exports it to parquet with automatic backups.
    
    master = MasterManager(config)
    master.export_ror_master()

    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        base = config["io"]["base_uri"]
        # self.master_uri = os.path.join(base, config["paths"]["master"]["path"])
        self.master_uri = f"{base}{config['paths']['master']['path']}"
        self.backup_master_dir = config["io"]["backup_master_uri"]
        # self.ror_uri = os.path.join(base, config["paths"]["ror"]["path"])
        self.ror_uri = f"{base}{config['paths']['ror']['path']}"
        self.df_ror_json = pd.read_json(self.ror_uri)
        self.df_ror_json_filtered = self.reshape_ror()
         # then dedupe + add uuid + match_field
        self.df_dedup_ror_json_filtered = self.dedupe_and_prepare_ror()
  
    def uuid_key(self, df, key_cols=['id_ror','ror_display_name', 'state_region', 'city','country']):
    
        DELIM = '\x1f'

        key_cols = key_cols
        norm = (
            df.loc[:, key_cols]
            .astype("string")          
            .fillna("")
            .stack()                   
            .str.normalize("NFKC")    
            .str.strip()
            .str.casefold()
            # back to the original shape
            .unstack(fill_value="")     
            .astype("string")
        )

        joined = norm.agg(DELIM.join, axis=1)
        # hash id
        # df_master['id'] = joined.map(lambda s: hashlib.md5(s.encode('utf-8')).hexdigest())

        df['id_uuid'] = joined.map(lambda s: str(uuid5(NAMESPACE_URL, s)))
        
        return df
  
    def match_field_col(self, df_ror_filtered):
        
        # concat: 'affiliations_clean' used for mapping to ROR-a bit weird as had to solve for NAs 
        s_org  = df_ror_filtered["ror_display_name"].astype("string").str.strip()
        s_city = df_ror_filtered["city"].astype("string").str.strip()
        s_reg  = df_ror_filtered["state_region"].astype("string").str.strip()
        s_ctry = df_ror_filtered["country"].astype("string").str.strip()

        def add_part(base, part):
            """
            Combine two string Series:
            - If both base and part are non-empty -> 'base, part'
            - If part is empty -> keep base
            - If base is empty -> just part
            NaNs are treated as empty strings.
            """
            base = base.fillna("")
            part = part.fillna("")
            return base + np.where((base != "") & (part != ""), ", " + part, part)

        aff = add_part(add_part(add_part(s_org, s_city), s_reg), s_ctry)
        df_ror_filtered["match_field"] = aff.mask(aff.str.strip().eq(""), pd.NA)
        
        return df_ror_filtered
  
    def reshape_ror(self):
        
             
        ################################################### 
        # names fields extract
        ###################################################
        df_ror_json_names = (
            self.df_ror_json
            .filter(['id', 'names'])
            .explode('names', ignore_index=True)
            .dropna(subset=['names'])
            .assign(
                types=lambda d: d['names'].str.get('types'),
                value=lambda d: d['names'].str.get('value')
            )[["id", "types", "value"]]
            .explode('types')
            .groupby(["id", "types"])
            .agg(list)
            .reset_index()
            .pivot(index="id", columns="types", values="value")
            .reset_index()
        )

        df_ror_json_names["ror_display_name"] = df_ror_json_names["ror_display"].str[0].str.strip()

        # strip duplicated ror display values from other fields
        for col in ["acronym", "alias", "label"]:
            df_update = (
                df_ror_json_names
                .explode(col)
                .filter(["id", col, "ror_display_name"])
                .loc[lambda _df: _df[col] != _df["ror_display_name"]]
                .groupby("id", sort=False)
                .agg(list)
                .reset_index()
                .filter(["id", col])
            )
            
            df_update = df_update[df_update[col].str[0].notna()]
            
            df_ror_json_names = (
                df_ror_json_names
                .drop(col, axis="columns")
                .merge(df_update, on="id", how="left")
            )

        df_ror_json_names = (
            df_ror_json_names
            .rename(columns={'label': 'names_lang'})
            .drop("ror_display", axis="columns")
        )

        assert df_ror_json_names[df_ror_json_names["id"].duplicated(keep=False)].empty

        ################################################## 
        # links field extract (links.type.website)
        ################################################## 

        df_ror_json_links = (
            self.df_ror_json
            .filter(['id', 'links'])
            .explode('links', ignore_index=True)
            .dropna(subset=['links'])
            .assign(
                types=lambda d: d['links'].str.get('type'),
                value=lambda d: d['links'].str.get('value')
            )[["id", "types", "value"]]
            .loc[lambda _df: _df["types"] == "website"]
            .explode('types')
            .groupby(["id", "types"])
            .agg(list)
            .reset_index()
            .filter(["id", "value"])
            .rename(columns={"value": "website_url"})
        )

        assert df_ror_json_links[df_ror_json_links["id"].duplicated(keep=False)].empty

        ################################################### 
        # locations field extract 
        ################################################### 
        df_ror_json_locations = (
            self.df_ror_json
            .filter(['id', 'locations'])
            .explode('locations', ignore_index=True)
            .dropna(subset=['locations'])
            .assign(
                country_code=lambda _df: _df['locations'].str.get("geonames_details").str.get('country_code'),
                country=lambda _df: _df['locations'].str.get("geonames_details").str.get('country_name'),
                state_region_code=lambda _df: _df['locations'].str.get("geonames_details").str.get('country_subdivision_code'),
                state_region=lambda _df: _df['locations'].str.get("geonames_details").str.get('country_subdivision_name'),
                city=lambda _df: _df['locations'].str.get("geonames_details").str.get('name')
            )
            .drop_duplicates(subset=["id"], keep="first")
            .drop("locations", axis="columns")
        )

        assert df_ror_json_locations[df_ror_json_locations["id"].duplicated(keep=False)].empty


        df_ror_json_filtered = (
            self.df_ror_json
            .filter(["id", "relationships", "ror_display_lang", "status", "types"])
            .merge(df_ror_json_names,on="id", how="inner")
            .merge(df_ror_json_links, how="left")
            .merge(df_ror_json_locations, on="id", how="inner")
            .rename(columns={"id": "id_ror"})
            .loc[lambda _df: _df["status"] != "withdrawn"]
            .filter([
                'id_ror',
                'ror_display_name',
                'alias',
                'names_lang',
                'relationships',
                'website_url',
                'city',
                'state_region_code',
                'state_region',
                'ror_display_lang', 
                'country',
                'country_code',
                'acronym',
                'status',
                'types'
            ])
        )

        return df_ror_json_filtered
       
    def dedupe_and_prepare_ror(self):
        ################################################### 
        # dedup on subset=['ror_display_name',  'city', 
        #                     'state_region', 'country']
        ################################################### 
        
        # dedupe Université Jean Monnet and Université de Bretagne Occidentale keeping active and dropping inactive 
        dupes_status = self.df_ror_json_filtered[self.df_ror_json_filtered.duplicated(subset=['ror_display_name',  'city', 
                                                                        'state_region', 'country'], 
                                                                keep=False)]
        df_dedup_ror_json_filtered = self.df_ror_json_filtered[~self.df_ror_json_filtered['id_ror'].isin(
            dupes_status.loc[dupes_status['status'].str.lower().isin(['withdrawn','inactive']), 'id_ror']
        )].copy()

        # dedupe Razi Hospital appears to be mistake with address
        df_dedup_ror_json_filtered.loc[df_dedup_ror_json_filtered['id_ror'] == 'https://ror.org/05rteaz10',
                            ['state_region', 'city','state_region_code']] = ['Gilan', 'Rasht', np.nan],

        assert not df_dedup_ror_json_filtered.duplicated(
            subset=['ror_display_name', 'city', 'state_region', 'country'],
            keep=False,
        ).any()

        # create uuid_key
        df_dedup_ror_json_filtered = self.uuid_key(df_dedup_ror_json_filtered)
        # create matched field
        df_dedup_ror_json_filtered = self.match_field_col(df_dedup_ror_json_filtered)
        
        self.df_dedup_ror_json_filtered = df_dedup_ror_json_filtered
        
        return df_dedup_ror_json_filtered

    def export_ror_master(self):
        
            # Backup old parquet, then overwrite
        try:
            df_master = pd.read_parquet(self.master_uri)

            ts = datetime.now().strftime("%Y%m%dT%H%M%S")
            backup_path = f"{self.backup_master_dir}/master_{ts}.parquet.gz"

            df_master.to_parquet(backup_path, compression="gzip")
            print(f"master file backup exported: {backup_path}")

        except FileNotFoundError:
            # first run / no existing master file
            print(f"No existing master file at {self.master_uri}, skipping backup.")

        (
            self.df_dedup_ror_json_filtered
            .rename(columns={"ror_display_name": "name"})
            .to_parquet(self.master_uri, compression="gzip")
        )
        print(f"ror exported to parquet: {self.master_uri}")
        
class ArtificialMappings:
    def __init__(self, df_ror: pd.DataFrame):
        self.df_ror = df_ror
        

    def make_display_name_combo_mapping(self, fields: list[str]):
        
        # ***
        # fields = empty list [] if display name only
        
        # cols = 'id_uuid', 'id_ror', affiliation_clean
 
        # deduplicates 
        # 'ror_display_name' 
        # 'ror_display_name' + 'country' 
        # 'ror_display_name' + 'city' 
        # 'ror_display_name' + 'state_region' 
        # 'ror_display_name' + 'state_region'  + 'city' 
        # 'ror_display_name'+ 'state_region' + 'country'  

        # no deduping needed
        # matched field ('ror_display_name' + 'city'+ 'state_region'+ 'country' )
        #  'ror_display_name' + 'city' + 'country' 
        # 'ror_display_name' + 'city'  + 'state_region' 
        
        # additional names 

        # ***
        
        subset = ['name'] + fields
        cols = ['id_ror', 'id_uuid', 'name'] + fields + ['match_field']
        
        # rows where (display_name, country_name) appears exactly once   
        unique_mask = (
            self.df_ror['name'].notna()
            & self.df_ror[fields].notna().all(axis=1)
            & ~self.df_ror.duplicated(subset=subset, keep=False)
        )

        df_mappings_display_name_fields = (
            self.df_ror.loc[unique_mask, cols]
            .reset_index(drop=True)
            .rename(columns={'name': 'affiliation_clean'})
            .assign(link_ROR='link_all')
        )

        # rows whose display name country are duplicated
        dupe_mask = (
            self.df_ror['name'].notna()
            & self.df_ror[fields].notna().all(axis=1)
            & self.df_ror.duplicated(subset=subset, keep=False)
        )

        df_mappings_display_name_fields_deduped = (
            self.df_ror.loc[dupe_mask, cols]
            .drop_duplicates(subset=subset)
            .reset_index(drop=True)
            .rename(columns={'name':'affiliation_clean'})
            .assign(link_ROR='name_' + "_".join(fields) + '_only')
        )
        
        # materialise the slices (actual dfs of the matching rows)
        self.df_unique = self.df_ror.loc[unique_mask, cols].copy()
        self.df_dupes  = self.df_ror.loc[dupe_mask,  cols].copy()

        df_mappings_display_name_fields_concat = pd.concat([
            df_mappings_display_name_fields, 
            df_mappings_display_name_fields_deduped
        ])
        
        cols_for_affiliation = ['affiliation_clean'] + fields
        df_mappings_display_name_fields_concat["affiliation_clean"] = (
            df_mappings_display_name_fields_concat[cols_for_affiliation]
            .fillna('')
            .agg(', '.join, axis=1)
            # .str.replace(r'(,\s*)+', ', ', regex=True)  # collapse ", , "
            .str.strip(', ')
        )
        
        # df_mappings_display_name_fields_concat.drop(
        #     columns=[c for c in fields if c != "country"],
        # inplace=True
        # )
        
        df_mappings_display_name_fields_concat.drop(
            columns=fields,
            inplace=True
        )

        df_mappings_display_name_fields_concat['date_added'] = date.today()
        
        return df_mappings_display_name_fields_concat
    
    
class Evaluation:
    """
   needs up dating will break  TODO needs up dating will break 
    Evaluator:
    - Reads a ground-truth file with ['affiliation_clean','id_mapping_true'].
    - Read an empty mapping file (no matches)
    - Builds a 'mapping_unified' from that truth (validated='n').
    - Uses SemanticMatcher.apply_match(...) to get predictions.
    - Merges and computes accuracy + a few metrics.
    Results are available on self.df_eval and self.report.
    """

    def __init__(self, config, build=False):
        self.config = config
        base = config["io"]["base_uri"]

        # Load ground truth
        truth_path  = base + config["paths"]["input"]["path"]
        self.df_ground_truth = (
            pd.read_excel(truth_path)
              .rename(columns=str.strip)
              .assign(
                  affiliation_clean=lambda d: d["affiliation_clean"].astype(str).str.lower().str.strip(),
                  id_mapping_true=lambda d: d["id_mapping_true"].astype("string")
              )
              .loc[:, ["affiliation_clean","id_mapping_true"]]
        )

        #  Build a mapping_unified *to be matched* (one row per affiliation, validated='n')
        mapping_unified = (
            self.df_ground_truth[["affiliation_clean"]].drop_duplicates()
              .assign(
                  id_mapping=pd.NA,
                  validated="n",
                  match_field=pd.NA,
                  matched_name=pd.NA,
                  matched_country=pd.NA,
                  match_score=pd.NA,
                  date_added=pd.NA
              )
        )

      
        master_uri = base + config["paths"]["master"]["path"]
        sm = SemanticMatcher(master_uri=master_uri, mapping_unified=mapping_unified, config=config)


        if build:
            sm.build()

        # Predict (matcher returns id_mapping + match_score per affiliation that passed threshold 
        # - note need to make sure threahold passes all as dont want to bias results)
        df_preds = sm.apply_match(mapping_unified=mapping_unified).rename(columns={
            "id_mapping": "id_mapping_pred"
        })[["affiliation_clean","id_mapping_pred","match_score"]]

        # Merge, score
        self.df_eval = (
            self.df_ground_truth.merge(df_preds, on="affiliation_clean", how="left")
               .assign(
                   id_mapping_pred=lambda d: d["id_mapping_pred"].astype("string"),
                   correct=lambda d: d["id_mapping_pred"] == d["id_mapping_true"]
               )
        )

        # Metrics
        mask_truth = self.df_eval["id_mapping_true"].notna()
        mask_pred  = self.df_eval["id_mapping_pred"].notna()
        mask = mask_truth & mask_pred                       # only rows with BOTH
        # coverage: how many truth rows actually received a prediction
        self.coverage = float(mask.sum() / max(1, mask_truth.sum()))
        
        bins = np.arange(0, 1.01, 0.1).tolist() + [float("inf")]
        self.df_eval["score_bucket"] = pd.cut(self.df_eval["match_score"], bins=bins)

        y_true = self.df_eval.loc[mask, "id_mapping_true"].astype("string")
        y_pred = self.df_eval.loc[mask, "id_mapping_pred"].astype("string")


        self.accuracy_by_score_bucket = (self.df_eval.loc[mask]
                                         .groupby("score_bucket", observed=True)["correct"]
                                         .mean()
                                         )

    
        cr = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        self.report =  pd.DataFrame.from_dict(
            # including accuracy caused a bug
            {k: v for k, v in cr.items() if k != "accuracy"},
            orient="index"
            )
        self.report_summary = {
            "n": int(mask.sum()),
            "coverage": self.coverage,                     
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": cr["macro avg"]["f1-score"],
            "weighted_f1": cr["weighted avg"]["f1-score"],
            "macro_precision": cr["macro avg"]["precision"],
            "macro_recall": cr["macro avg"]["recall"],
            "weighted_precision": cr["weighted avg"]["precision"],
            "weighted_recall": cr["weighted avg"]["recall"],
        }
   
         

   