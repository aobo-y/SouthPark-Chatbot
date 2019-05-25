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


class Vocabulary(EmbeddingMap):
    '''
    embedding maps for words
    '''

    pad = '<pad>' # Padding token
    sos = '<sos>' # Start of Sentence token
    eos = '<eos>' # End of Sentence token
    unk = '<unk>'  # pretrained word embedding usually has this

    def __init__(self, tokens=[]):
        super().__init__(tokens)

        for special_token in [self.pad, self.sos, self.eos, self.unk]:
            if not self.has(special_token):
                self.append(special_token)

        self.pad_idx = self.get_index(self.pad)
        self.sos_idx = self.get_index(self.sos)
        self.eos_idx = self.get_index(self.eos)
        self.unk_idx = self.get_index(self.unk)


class Persons(EmbeddingMap):
    '''
    embedding maps for persona
    '''

    none = '<none>'

    def __init__(self, tokens=[]):
        super().__init__(tokens)

        self.append(self.none)
        self.none_idx = self.get_index(self.none)
