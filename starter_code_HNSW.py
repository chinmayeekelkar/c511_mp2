import faiss
import h5py
import numpy as np
import os
import requests

def evaluate_hnsw():

    # start your code here
    # download data, build index, run query

    # Download SIFT1M dataset from ann-benchmarks
    sift_url = "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    sift_filename = "sift-128-euclidean.hdf5"
    
    if not os.path.exists(sift_filename):
        print("Downloading SIFT dataset...")
        response = requests.get(sift_url, stream=True)
        response.raise_for_status()
        
        with open(sift_filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("SIFT dataset downloaded!")
    else:
        print("SIFT dataset already exists")

    # Load the dataset from HDF5 format
    print("Loading dataset...")
    with h5py.File(sift_filename, 'r') as f:
        database_vectors = f['train'][:]  # 1M database vectors
        query_vectors = f['test'][:]      # Query vectors
    
    print(f"Database vectors shape: {database_vectors.shape}")
    print(f"Query vectors shape: {query_vectors.shape}")
    
    # Create HNSW index with specified parameters
    # M=16, efConstruction=200, efSearch=200
    print("Creating HNSW index...")
    dimension = database_vectors.shape[1]  # 128 dimensions
    index = faiss.IndexHNSWFlat(dimension, 16)  # M=16
    index.hnsw.efConstruction = 200
    index.hnsw.efSearch = 200
    
    # Add database vectors to index
    print("Adding database vectors to index...")
    index.add(database_vectors.astype(np.float32))
    
    # Perform query using first query vector
    print("Performing query...")
    first_query = query_vectors[0:1].astype(np.float32)  # Use first query vector
    
    # Search for top 10 nearest neighbors
    k = 10
    distances, indices = index.search(first_query, k)
    
    print(f"Top 10 nearest neighbor indices: {indices[0]}")
    
    # write the indices of the 10 approximate nearest neighbours in output.txt, separated by new line in the same directory
    print("Writing results to output.txt...")
    with open("output.txt", "w") as f:
        for idx in indices[0]:
            f.write(f"{idx}\n")
    
    print("Done! Results written to output.txt")

if __name__ == "__main__":
    evaluate_hnsw()
