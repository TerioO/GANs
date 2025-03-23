import { defineStore } from "pinia";
import { useFetchWrapper } from "../hooks/useFetchWrapper";
import type { IMsgResponse, IOnnxRequest } from "../types/api-types";
import axios from "axios";

const baseURL = import.meta.env.VITE_API_BASE_URL;

export const useApiStore = defineStore("api", () => {
  const baseQuery = axios.create({
    baseURL,
    headers: {
      "Content-Type": "application/json"
    }
  });

  const getServerStatus = useFetchWrapper<IMsgResponse>(() => {
    return baseQuery.get("/api/server-status");
  });

  const getGanSimpleV4 = useFetchWrapper<IOnnxRequest["data"], IOnnxRequest["payload"]>(
    (payload) => {
      return baseQuery.get(`/api/getGanSimpleV4Images?batchSize=${payload["batchSize"]}`);
    }
  );

  return {
    baseQuery,
    getServerStatus,
    getGanSimpleV4
  };
});
