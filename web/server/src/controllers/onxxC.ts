import { Request, Response, NextFunction } from "express";
import { InferenceSession, Tensor } from "onnxruntime-node";
import { rootDir } from "../paths";
import path from "path";
import gaussian from "gaussian";
import { IOnnxRequest } from "../types/global-types";
import createHttpError from "http-errors";

export const getGanSimpleV4Images = async (
    req: Request<unknown, unknown, unknown, IOnnxRequest["payload"]>,
    res: Response,
    next: NextFunction
) => {
    try {
        const batchSize = parseInt(req.query.batchSize);
        if (!batchSize) throw createHttpError(400, "batchSize must be a number");
        if (batchSize < 1 || batchSize > 64)
            throw createHttpError(400, "batchSize must be in this interval: [1,64]");

        const file = path.join(rootDir, "assets", "GAN_simple_v4.onnx");
        const session = await InferenceSession.create(file);

        const distribution = gaussian(0, 1);

        const imgSize = 28;
        const x = new Float32Array(batchSize * imgSize * imgSize);
        for (let i = 0; i < x.length; i++) {
            x[i] = distribution.ppf(Math.random());
        }

        const input = new Tensor("float32", x, [batchSize, 1, imgSize, imgSize]);
        const results = await session.run({ input });

        const dims = results.output.dims;

        const denormalized: any[] = [];
        for (let i = 0; i < results.output.data.length; i++) {
            const value = results.output.data[i] as number;
            denormalized[i] = Math.floor((value + 1) * 127.5);
        }

        const data: IOnnxRequest["data"] = {
            tensor: denormalized,
            dims
        };

        res.status(200).json(data);
    } catch (error) {
        if (createHttpError.isHttpError(error)) next(error);
        else next(createHttpError(500, "Failed to run model"));
    }
};
