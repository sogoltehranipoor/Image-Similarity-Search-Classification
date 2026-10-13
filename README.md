# Image-Similarity-Search-Classification
Upload an image and get back **what it is** plus **the most similar images** from a reference library. No model training required: it uses a pretrained MobileNetV2 as a feature extractor and a weighted k-nearest-neighbour vote.

1. **Data**: Caltech256 (downloaded with `kagglehub`). The `257.clutter` class is skipped, leaving 256 real object categories.
2. **Library**: 60 random images per class are indexed. 2 extra images per class are held out for fair evaluation.
3. **Features**: MobileNetV2 (ImageNet weights, `include_top=False`, average pooling) turns each image into a 1280-dimensional vector. Vectors are L2-normalised, so a dot product equals cosine similarity.
4. **Prediction**: the query vector is compared to every library vector. The top 15 most similar images vote for a category, each vote weighted by its similarity.
5. **Retrieval**: the 8 most similar images from the winning category are displayed next to the query.

## Evaluation

| Method | What it measures |
|---|---|
| `category_accuracy` | 500 random library images, each queried against the rest (the image itself is excluded). Slightly optimistic. |
| `heldout_accuracy` | Images never added to the library. The more realistic number. |

Results: _add your numbers here after running the notebook_.

## Run it

Open the notebook in Google Colab (GPU runtime recommended: Runtime → Change runtime type → GPU) and run the cells in order. The last cell lets you upload your own image.

```bash
pip install tensorflow kagglehub matplotlib numpy
```

## Project structure

```
.
├── image_similarity_search.ipynb   # the full pipeline
├── requirements.txt
└── README.md
```

## Notes

- Feature extraction over the whole library is the slowest step. `features.npy` and `labels.npy` are saved so it only needs to run once.
- Possible improvements: approximate nearest neighbours (FAISS), fine-tuning the backbone, a stronger model such as EfficientNet or CLIP, and a small web UI (Streamlit).

## Credits

- Dataset: [Caltech256](https://www.kaggle.com/datasets/jessicali9530/caltech256) (Griffin, Holub, Perona).
- Model: MobileNetV2 pretrained on ImageNet, via `tf.keras.applications`.
