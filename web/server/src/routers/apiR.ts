import express from "express";
import * as api from "../controllers/apiC";

const router = express.Router();

router.get("/api/server-status", api.getServerStatus);

export default router;
