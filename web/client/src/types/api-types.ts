export interface IApiError {
    message: string;
    isError: boolean;
}

export interface IMsgResponse {
    message: string;
}

export type TOnnxGanNames = "GAN_simple_v4" | "DCGAN_MNIST_v1" | "DCGAN_Cats_v0";
export type TOnnxCganNames =
    | "CDCGAN_MNIST_v3"
    | "CDCGAN_Cats_v1"
    | "CDCGAN_Animal_Faces_v7"
    | "CDCGAN_FashionMNIST_v2";

interface GanModelData {
    imgSize: number;
    outShape: number[];
    inShape: number[];
    maxBatchSize: {
      dev: number;
      prod: number;
    }
}

interface CganModelData extends GanModelData {
    numClasses: number;
    classes: {
        [key: number]: string;
    };
}

export type TGanModel = Record<TOnnxGanNames, GanModelData>;
export type TCganModel = Record<TOnnxCganNames, CganModelData>;

export interface IOnnxGanRequest {
    payload: {
        batchSize: number;
        modelName: TOnnxGanNames;
    };
    res: {
        tensor: any[];
        dims: readonly number[];
    } & Omit<GanModelData, "maxBatchSize">;
}

export interface IOnnxCganRequest {
    payload: {
        batchSize: number;
        label: number;
        modelName: TOnnxCganNames;
    };
    res: {
        tensor: any[];
        dims: readonly number[];
    } & Omit<CganModelData, "maxBatchSize">;
}
