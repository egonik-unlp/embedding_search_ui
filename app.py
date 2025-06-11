import streamlit as st
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

client = QdrantClient(
 	url=st.secrets["qdrant"]["url"],
 	api_key=st.secrets["qdrant"]["key"]
 )

# client = QdrantClient(url="http://qdrant:6333")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# model = model.to("cpu")
_ = model.encode("test", normalize_embeddings=True) 

def busqueda_metodo_anterior(documents):
	return "todo"

def busqueda(documents):
	for query in documents:
		print(query)
		if query != "":
			query_embedding = model.encode(query, normalize_embeddings=True)

			# Perform search
			results = client.search(
			collection_name="ingredientes_vec2",
			query_vector=query_embedding.tolist(), 
			limit=1
		)
			
		for result in results:
			res = result.payload, f"query => {query}→ Score: {result.score:.3f}"
			x = result.payload
			x["query"] = query # type: ignore
			x["score"] = result.score
			st.json(result.payload)
			print(res)

def busqueda_mas75(documents):
	for query in documents:
		print(query)
		if query != "":
			query_embedding = model.encode(query, normalize_embeddings=True)

			# Perform search
			results = client.search(
			collection_name="ingredientes_vec2",
			query_vector=query_embedding.tolist(), 
			limit=1
		)

	
		for result in results:
			res = result.payload, f"query => {query}→ Score: {result.score:.3f}"
			x = result.payload
			x["query"] = query
			x["score"] = result.score
			if result.score > 0.60:
				st.json(result.payload)
				print(res)

def main():
	st.title('Busqueda GV 2.0')
	query = st.text_area("Escribir terminos de busqueda separados por ','" )
	documents = query.split(",")
	if st.button("busqueda"):
		busqueda(documents)
	if st.button("busqueda mayor similitud"):
		busqueda_mas75(documents)


if __name__ == "__main__":
	main()
