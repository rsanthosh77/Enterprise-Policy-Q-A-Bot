"""Dense embeddings hosted by Pinecone; no additional provider key required."""
class PineconeEmbeddings:
    def __init__(self, client, model='llama-text-embed-v2', dimensions=1024):
        self.client = client
        self.model = model
        self.dimensions = dimensions

    def _embed(self, texts, input_type):
        vectors = []
        # llama-text-embed-v2 supports batches of up to 96; leave headroom.
        for offset in range(0, len(texts), 64):
            batch = texts[offset:offset + 64]
            result = self.client.inference.embed(
                model=self.model, inputs=batch,
                parameters={'input_type': input_type, 'dimension': self.dimensions, 'truncate': 'NONE'})
            batch_vectors = [list(item.values) for item in result.data]
            if len(batch_vectors) != len(batch) or any(len(v) != self.dimensions for v in batch_vectors):
                raise ValueError('Pinecone returned unexpected embedding count or dimensions; snapshot not activated.')
            vectors.extend(batch_vectors)
        return vectors

    def embed_documents(self, texts):
        return self._embed(texts, 'passage')

    def embed_query(self, text):
        return self._embed([text], 'query')[0]
