import { useFetchWrapper } from "../hooks/useFetchWrapper";
import type { IOnnxCganRequest, IMsgResponse, IOnnxGanRequest } from "../types/api-types";
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
  useFetchWrapper<IOnnxGanRequest["res"], IOnnxGanRequest["payload"]>((payload) => {
    return baseQuery.post("/api/run-gan", payload);
  });

export const runCGAN = () =>
  useFetchWrapper<IOnnxCganRequest["res"], IOnnxCganRequest["payload"]>((payload) => {
    return baseQuery.post("/api/run-cgan", payload);
  });
