import type { TOnnxModel, TOnnxModelName } from "./global-types";

export const onnxModelNames: TOnnxModelName[] = [
    "DCGAN_MNIST_v0", 
    "GAN_simple_v4",
    "DCGAN_Cats_v0"
] as const;

export const onnxModels: TOnnxModel = {
    "GAN_simple_v4": {
        imgSize: 28,
        inShape: [1, 1, 28, 28],
        outShape: [1, 1, 28, 28]
    },
    "DCGAN_MNIST_v0": {
        imgSize: 28,
        inShape: [1, 100, 1, 1],
        outShape: [1, 1, 28, 28]
    },
    "DCGAN_Cats_v0": {
        imgSize: 64,
        inShape: [1, 100, 1, 1],
        outShape: [1, 3, 64, 64]
    }
};