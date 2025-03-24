import express from "express";
import * as api from "../controllers/apiC";
import * as onnxC from "../controllers/onxxC";

const router = express.Router();

router.get("/api/server-status", api.getServerStatus);
router.post("/api/run-gan", onnxC.runGAN);

export default router;
