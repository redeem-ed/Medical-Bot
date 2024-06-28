## Some useful links: 

**Important links:**
  
> [Hugging Face Quantized model by bloke](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/blob/main/llama-2-7b-chat.ggmlv3.q8_0.bin)

> [Ctransformers](https://github.com/marella/ctransformers)

> [Meta](https://huggingface.co/meta-llama)

> [Chainlit](https://docs.chainlit.io/get-started/overview)






1. Need llama 2 model that is Quantized (put in bits 1bit, 4 or 8 bits). This is provided by Bloke
2. to run it on our machine we need CTransformers (Python binding in c/c++) 
3. will use sentence Transformers to create embeddings. 
	1. Now, we have created embeddings, we need some kind of vector database or vector store to cast this embeddings. 
    * The most famous is ChromaDB
		* Faiss CPU (will use this version but they also have GPU) 
		* PineCone (it is closed source) 
		* Qdrant 

High level information: 
Docs Data → Pre-process (this is done by langchain loaders, splitters and many more loaders) → pass this to an embedding model (Sentence Transformers) → Save this in Vector DB (Faiss CPU) 

Vector stores features: 
- it has in built similarity algorithms like cosine algorithm 
- they're faster therefore no latency issue as well 
- metadata etc. 
