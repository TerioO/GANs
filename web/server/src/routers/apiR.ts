import express from "express";
import * as api from "../controllers/apiC";
import * as onnxC from "../controllers/onxxC";

const router = express.Router();

router.get("/api/server-status", api.getServerStatus);
router.get("/api/getGanSimpleV4Images", onnxC.getGanSimpleV4Images);

export default router;
