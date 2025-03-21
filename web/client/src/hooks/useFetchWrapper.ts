import { ref } from "vue";
import { AxiosError, type AxiosResponse } from "axios";
import type { IApiError } from "../types/api-types";

interface Data<T> extends AxiosResponse<T> {
    ok: boolean;
}

export function useFetchWrapper<T>(request: () => Promise<AxiosResponse<T>>) {
    const loading = ref<boolean>(false);
    const isErr = ref<boolean>(false);
    const errMsg = ref<string | null>(null);
    const data = ref<T | null>(null);
    const axiosRes = ref<Data<T> | null>(null);

    const trigger = async function () {
        try {
            loading.value = true;
            const res = await request();
            axiosRes.value = {
                ...res,
                ok: res.status > 200 && res.status <= 299 ? true : false
            };
            data.value = res.data;
        } catch (error) {
            if (error instanceof AxiosError) {
                const apiError = error as AxiosError<IApiError>;
                if (!apiError.response) return;
                isErr.value = apiError.response.data.isError;
                errMsg.value = apiError.response.data.message;
            } else if (error instanceof Error) {
                errMsg.value = "Unknown error";
                isErr.value = true;
            }
        } finally {
            loading.value = false;
        }
    };

    return { loading, isErr, errMsg, data, axiosRes, trigger };
}
