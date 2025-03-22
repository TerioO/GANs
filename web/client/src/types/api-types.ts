export interface IApiError {
    message: string;
    isError: boolean;
}

export interface IMsgResponse {
    message: string;
}

export interface IOnnxRequest {
    payload: {
        batchSize: string;
    };
    data: {
        tensor: any[];
        dims: readonly number[];
    }
}