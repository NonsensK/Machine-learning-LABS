# Fashion-MNIST data card

| Field | Value |
| --- | --- |
| Source | [Zalando Research Fashion-MNIST](https://github.com/zalandoresearch/fashion-mnist) |
| Image format | 28 × 28 grayscale pixels |
| Task | Ten-class clothing classification |
| Official split | 60,000 training images and 10,000 test images |
| Lab 6 split | Seeded 55,000/5,000 split of the official training set |
| Transform | Convert to a tensor and normalize with mean 0.2860 and standard deviation 0.3530 |
| Lab metrics | Validation loss and accuracy |
| Project evaluation | Official test split, reserved for final evaluation |

The classes are T-shirt/top, trouser, pullover, dress, coat, sandal, shirt, sneaker, bag and ankle boot. The benchmark uses small grayscale product images. Results describe this benchmark and may not transfer to photographs with backgrounds, different lighting or multiple garments.
