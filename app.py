import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""




client = QdrantClient(
 	url=st.secrets["qdrant"]["url"],
 	api_key=st.secrets["qdrant"]["key"]
 )

#client = QdrantClient(
#	host = "localhost",
#	port= 6333	
#)
model = SentenceTransformer("sentence-transformers/multi-qa-MiniLM-L6-cos-v1")
# model_name = "sentence-transformers/all-MiniLM-L6-v2"




model = model.to("cpu")
_ = model.encode("test", normalize_embeddings=True) 
def busqueda(documents):
	for query in documents:
		print(query)
		if query != "":
			query_embedding = model.encode(query, normalize_embeddings=True)

			# Perform search
			results = client.search(
			collection_name="ingredientes",
			query_vector=query_embedding.tolist(), 
			limit=1
		)
			
		for result in results:
			res = result.payload, f"query => {query}→ Score: {result.score:.3f}"
			x = result.payload
			x["query"] = query
			x["score"] = result.score
			st.json(result.payload)
			print(res)

def busqueda_mas75(documents):
	for query in documents:
		if query != "":
			query_embedding = model.encode(query, normalize_embeddings=True)

			# Perform search
			results = client.search(
			collection_name="ingredientes",
			query_vector=query_embedding.tolist(), 
			limit=1
		)

	
		for result in results:
			res = result.payload, f"query => {query}→ Score: {result.score:.3f}"
			x = result.payload
			x["query"] = query
			x["score"] = result.score
			if result.score > 0.75:
				st.json(result.payload)
				print(res)


st.title('Busqueda GV 2.0')

# query = st.text_input()

query = st.text_area("Escribir terminos de busqueda separados por ','" )
documents = query.split(",")
if st.button("busqueda"):
	busqueda(documents)
if st.button("busqueda mayor similitud"):
	busqueda_mas75(documents)



