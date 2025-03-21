import { defineStore } from "pinia";
import { useFetchWrapper } from "../hooks/useFetchWrapper";
import axios from "axios";
import type { IMsgResponse } from "../types/api-types";

const baseURL = import.meta.env.VITE_API_BASE_URL;

export const useApiStore = defineStore("api", () => {
    const baseQuery = axios.create({
        baseURL,
        headers: {
            "Content-Type": "application/json"
        }
    })

    const getServerStatus = useFetchWrapper<IMsgResponse>(() => {
        return baseQuery.get("/api/server-status")
    })

    return {
        baseQuery,
        getServerStatus
    }
})