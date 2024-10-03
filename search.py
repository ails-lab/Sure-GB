from angle_emb import AnglE, Prompts
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


class Search:

    def __init__(self, texts):
        # self.model = AnglE.from_pretrained('WhereIsAI/UAE-Large-V1', pooling_strategy='cls').cuda()
        # self.angle.set_prompt(prompt=Prompts.C)
        self.angle = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").to("cuda")
        self.texts = texts
        self.encode(texts)

    def search(self, text):
        vec = self.angle.encode([{"text": text}])
        sims = {}
        for i, cand_vec in enumerate(self.vecs):
            similarity = cosine_similarity(vec.reshape(1, -1) , cand_vec.reshape(1, -1))[0][0]
            sims[i] = similarity
        return [[k, v] for k, v in sorted(sims.items(), key=lambda item: item[1], reverse = True)]

    def encode(self, texts, save_to_texts = True):
        if save_to_texts:
            self.vecs = self.angle.encode([{"text": text} for text in texts])
        else:
            return self.angle.encode([{"text": text} for text in texts])

        


# import faiss                   
# import numpy as np

# class Search:

#     def __init__(self, mapping, size = 768):
#         self.mapping = mapping

#         self.text_to_index = {text: index for index, text in enumerate(self.mapping.keys())}
#         self.index_to_text = {index: text for text, index in self.text_to_index.items()}
#         # create indexing array
#         self.embeddings_array = np.zeros((len(self.text_to_index), size))
#         for index in range(len(mapping)):
#             text = self.index_to_text[index]
#             self.embeddings_array[index] = self.mapping[text]

#         self.embeddings_array = self.embeddings_array.astype('float32')
#         self.search_engine = faiss.IndexFlatL2(size)
#         self.search_engine.add(self.embeddings_array)

#     def search(self, embs, k_results = 5, with_scores = False):
#         embs = embs.astype('float32')
#         D, I = self.search_engine.search(embs, k_results)
#         resp = []
#         for row, score_row in zip(I, D):
#             r = []
#             for i, score in zip(row, score_row):
#                 if with_scores:
#                     r.append([self.index_to_text[i], score])
#                 else:
#                     r.append(self.index_to_text[i])
#             resp.append(r.copy())
#         return resp