import type {
    TCganModel,
    TGanModel,
    TOnnxCganNames,
    TOnnxGanNames
} from "./global-types";

export const onnxGanNames: TOnnxGanNames[] = [
    "DCGAN_MNIST_v0",
    "GAN_simple_v4",
    "DCGAN_Cats_v0"
] as const;

export const onnxCganNames: TOnnxCganNames[] = [
    "CDCGAN_MNIST_v0",
    "CDCGAN_Cats_v0",
    "CDCGAN_Animal_Faces_v4",
    "CDCGAN_FashionMNIST_v0"
] as const;

export const onnxGanModels: TGanModel = {
    GAN_simple_v4: {
        imgSize: 28,
        inShape: [1, 1, 28, 28],
        outShape: [1, 1, 28, 28]
    },
    DCGAN_MNIST_v0: {
        imgSize: 28,
        inShape: [1, 100, 1, 1],
        outShape: [1, 1, 28, 28]
    },
    DCGAN_Cats_v0: {
        imgSize: 64,
        inShape: [1, 100, 1, 1],
        outShape: [1, 3, 64, 64]
    }
};

export const onnxCganModels: TCganModel = {
    CDCGAN_MNIST_v0: {
        imgSize: 28,
        inShape: [1, 100, 1, 1],
        outShape: [1, 1, 28, 28],
        numClasses: 10,
        classes: {
            "0": "0 - zero",
            "1": "1 - one",
            "2": "2 - two",
            "3": "3 - three",
            "4": "4 - four",
            "5": "5 - five",
            "6": "6 - six",
            "7": "7 - seven",
            "8": "8 - eight",
            "9": "9 - nine"
        }
    },
    CDCGAN_Cats_v0: {
        imgSize: 64,
        inShape: [1, 100, 1, 1],
        outShape: [1, 3, 64, 64],
        numClasses: 2,
        classes: { "0": "cats", "1": "dogs" }
    },
    CDCGAN_Animal_Faces_v4: {
        imgSize: 128,
        inShape: [1, 100, 1, 1],
        outShape: [1, 3, 128, 128],
        numClasses: 3,
        classes: { "0": "cat", "1": "dog", "2": "wild" }
    },
    CDCGAN_FashionMNIST_v0: {
        imgSize: 28,
        inShape: [1, 100, 1, 1],
        outShape: [1, 1, 28, 28],
        numClasses: 10,
        classes: {}
    }
};
