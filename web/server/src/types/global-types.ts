export interface IApiError {
    message: string;
    isError: boolean;
}

export interface IMsgResponse {
    message: string;
}

export type TOnnxModelName = "GAN_simple_v4" | "DCGAN_MNIST_v0" | "DCGAN_Cats_v0";

interface ModelData {
    imgSize: number;
    outShape: number[];
    inShape: number[];
}

export type TOnnxModel = Record<TOnnxModelName, ModelData>;

export interface IOnnxRequest {
    payload: {
        batchSize: string;
        modelName: TOnnxModelName;
    };
    res: {
        tensor: any[];
        dims: readonly number[];
    } & ModelData;
}
