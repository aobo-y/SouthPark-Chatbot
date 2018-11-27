class EmbeddingMap:
    '''
    convert index to token and vice versa
    '''

    def __init__(self, tokens=[]):
        self.token_2_index = {}
        self.tokens = [t for t in tokens] # clone

        for index, token in enumerate(tokens):
            self.token_2_index[token] = index

    def get_token(self, index):
        return self.tokens[index]

    def get_index(self, token):
        return self.token_2_index[token]

    def append(self, token):
        self.tokens.append(token)
        self.token_2_index[token] = len(self.tokens) - 1

    def has(self, token):
        return token in self.token_2_index

    def size(self):
        return len(self.tokens)
