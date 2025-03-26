import { reactive } from "vue";
import { AxiosError, type AxiosResponse } from "axios";
import type { IApiError } from "../types/api-types";

interface Data<T> extends AxiosResponse<T> {
  ok: boolean;
}

interface Req<T> {
  loading: boolean;
  isErr: boolean;
  errMsg: string | null;
  data: T | null;
  axiosRes: Data<T> | null;
}

export function useFetchWrapper<T, Payload = void>(
  request: (payload: Payload) => Promise<AxiosResponse<T>>
) {
  const req: Req<T> = reactive({
    loading: false,
    isErr: false,
    errMsg: null,
    data: null,
    axiosRes: null
  })

  const trigger = async function (p: Payload) {
    try {
      req.loading = true;
      req.isErr = false;
      req.errMsg = null;
      req.data = null;
      req.axiosRes = null;

      const res = await request(p);
      req.axiosRes = {
        ...res,
        ok: res.status >= 200 && res.status <= 299 ? true : false
      };
      req.data = res.data;
      return req.axiosRes;
    } catch (error) {
      if (error instanceof AxiosError) {
        const apiError = error as AxiosError<IApiError>;
        if (!apiError.response) return;
        req.isErr = apiError.response.data.isError;
        req.errMsg = apiError.response.data.message;
      } else if (error instanceof Error) {
        req.errMsg = "Unknown error";
        req.isErr = true;
      }
    } finally {
      req.loading = false;
    }
  };

  return { req, trigger };
}
