import type { TCganModel, TGanModel, TOnnxCganNames, TOnnxGanNames } from "./global-types";

export const onnxGanNames: TOnnxGanNames[] = [
    "DCGAN_MNIST_v0", 
    "GAN_simple_v4",
    "DCGAN_Cats_v0"
] as const;

export const onnxCganNames: TOnnxCganNames[] = [
    "CDCGAN_MNIST_v0",
    "CDCGAN_Cats_v0"
] as const;

export const onnxGanModels: TGanModel = {
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

export const onnxCganModels: TCganModel = {
    "CDCGAN_MNIST_v0": {
        imgSize: 28,
        numClasses: 10,
        inShape: [1, 100, 1, 1],
        outShape: [1, 1, 28, 28]
    },
    "CDCGAN_Cats_v0": {
        imgSize: 64,
        numClasses: 2,
        inShape: [1, 100, 1, 1],
        outShape: [1, 3, 64, 64]
    }
};