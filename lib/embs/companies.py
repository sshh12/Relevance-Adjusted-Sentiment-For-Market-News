from keras.layers import Input, Embedding, Dense, Dot, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model, load_model
import plotly.express as px
import pandas as pd
import numpy as np
import random
import umap
import glob
import os


class AbstractEmb:

    TAG = 'abs'

    def __init__(self, art_exp_id, art_embs, sym_to_idx, sym_to_art_idxs, **kwargs):
        self.art_embs = art_embs
        self.sym_to_idx = sym_to_idx
        self.sym_to_art_idxs = sym_to_art_idxs
        self.args = kwargs
        self.exp_id = 'company-embs-{}-{}-{}-{}'.format(
            len(sym_to_idx), art_exp_id, 
            self.TAG, '-'.join([str(v) for v in kwargs.values()])
        )
        self.exp_name = 'Company Embeddings ({})'.format(self.TAG)
        self.figs = {}

    def prep(self):
        pass

    def bake_embs(self):
        raise NotImplementedError()

    def plot(self, label_name, labels, names):
        assert len(labels) == len(self.sym_to_idx)
        reducer = umap.UMAP()
        comp_rembs = reducer.fit_transform(self.comp_embs)
        df = pd.DataFrame({
            'x': comp_rembs[:, 0], 'y': comp_rembs[:, 1],
            label_name: labels,
            'name': names
        })
        self.figs[label_name] = px.scatter(df, x="x", y="y", color=label_name, hover_name='name', title=self.exp_name)
        self.figs[label_name].show()

    def save_all(self, folder='data'):
        fn_embs = os.path.join(folder, '{}.npy'.format(self.exp_id))
        np.save(fn_embs, self.comp_embs)
        if len(self.figs) > 0:
            for name, fig in self.figs.items():
                fn_fig = os.path.join(folder, '{}-{}.png'.format(self.exp_id, name.lower()))
                fig.write_image(fn_fig)
        # model already saved


class KerasDeep(AbstractEmb):

    TAG = 'keras'

    def _build_model(self):

        art_emb_size = self.art_embs.shape[1]
        latent_size = self.args['latent_size']
        post_emb_layers = self.args['post_emb_layers']
        
        input_symbol = Input(shape=(1,), name='symbol')
        input_art_emb = Input(shape=(art_emb_size,), name='art_emb')

        embedding_symbol = Embedding(
            input_dim=len(self.sym_to_idx), 
            output_dim=latent_size, 
            name='symbol_emb'
        )(input_symbol)
        embedding_symbol = Reshape((latent_size,))(embedding_symbol)

        embedding_art = Dense(latent_size, activation='relu')(input_art_emb)
        for _ in range(post_emb_layers):
            embedding_art = Dense(latent_size, activation='relu')(embedding_art)

        merged = Dot(normalize=True, axes=1)([embedding_symbol, embedding_art])
        merged = Reshape(target_shape=(1,))(merged)

        out = Dense(1, activation='sigmoid', name='similarity')(merged)
        model = Model(inputs=[input_symbol, input_art_emb], outputs=out)

        model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def _train(self):
        checkpoint = ModelCheckpoint(self.model_save_path, monitor='val_accuracy', verbose=1, save_best_only=True)
        early_stop = EarlyStopping(monitor='val_accuracy', patience=4)
        S, A, Y = self.dataset
        hist = self.model.fit(x=[S, A], y=Y, epochs=100, batch_size=16, validation_split=0.3, 
            callbacks=[checkpoint, early_stop])

    def prep(self):
        self.model = self._build_model()
        self.model_save_path = os.path.join('data', self.exp_id + '-{epoch:02d}-{val_accuracy:.2f}.h5')
        self.models_glob = os.path.join('data', self.exp_id + '-*.h5')

    def bake_embs(self):

        self._train()
        del self.model

        models = glob.glob(self.models_glob)
        best_model = max(models, key=lambda path: float(path.split('-')[-1].replace('.h5', '')))
        for model_fn in models:
            if model_fn != best_model:
                os.remove(model_fn)

        self.model = load_model(best_model)
        self.sym_emb_model = Model(
            inputs=self.model.get_layer('symbol').get_input_at(0), 
            outputs=self.model.get_layer('symbol_emb').get_output_at(0)
        )

        self.comp_embs = self.sym_emb_model.predict(
            np.array(list(self.sym_to_idx.values()))).squeeze()


EMBEDDINGS = [
    KerasDeep
]