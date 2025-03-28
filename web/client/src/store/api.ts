import { useFetchWrapper } from "../hooks/useFetchWrapper";
import type { IMsgResponse, IOnnxRequest } from "../types/api-types";
import axios from "axios";

const baseURL = import.meta.env.VITE_API_BASE_URL;

export const baseQuery = axios.create({
  baseURL,
  headers: {
    "Content-Type": "application/json"
  }
});

export const getServerStatus = () =>
  useFetchWrapper<IMsgResponse>(() => {
    return baseQuery.get("/api/server-status");
  });

export const runGAN = () =>
  useFetchWrapper<IOnnxRequest["res"], IOnnxRequest["payload"]>((payload) => {
    return baseQuery.post("/api/run-gan", payload);
  });
