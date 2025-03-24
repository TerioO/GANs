import { InferenceSession, Tensor } from "onnxruntime-node";
import { rootDir } from "../paths";
import type { TOnnxModel, TOnnxModelName, IOnnxRequest } from "../types/global-types";
import { onnxModels } from "../types/constant-types";
import gaussian from "gaussian";
import path from "path";

export async function evalOnnxModel(modelName: TOnnxModelName, batchSize: number){
    const file = path.join(rootDir, "assets", `${modelName}.onnx`);
    const model = onnxModels[modelName];
    const session = await InferenceSession.create(file);

    const randn = gaussian(0, 1);
    const x = new Float32Array(batchSize * model.inShape[1] * model.inShape[2] * model.inShape[3]);
    for(let i=0; i<x.length; i++){
        x[i] = randn.ppf(Math.random());
    }

    const inShape = [batchSize, model.inShape[1], model.inShape[2], model.inShape[3]];
    const input = new Tensor("float32", x, inShape);
    const results = await session.run({ input });
    const outShape = [batchSize, model.outShape[1], model.outShape[2], model.outShape[3]];

    const denormalized: any[] = [];
    for(let i=0; i<results.output.data.length; i++){
        const value = results.output.data[i] as number;
        denormalized[i] = Math.floor((value + 1) * 127.5);
    }

    const res: IOnnxRequest["body"] = {
        tensor: denormalized,
        dims: results.output.dims,
        inShape,
        outShape,
        imgSize: onnxModels[modelName].imgSize
    };

    return res;
}