import { InferenceSession, Tensor } from "onnxruntime-node";
import { rootDir } from "../paths";
import type { IOnnxCganRequest, IOnnxGanRequest, TOnnxCganNames, TOnnxGanNames } from "../types/global-types";
import { onnxCganModels, onnxGanModels } from "../types/constant-types";
import gaussian from "gaussian";
import path from "path";

export async function evalOnnxModelGAN(modelName: TOnnxGanNames, batchSize: number){
    const file = path.join(rootDir, "assets", `${modelName}.onnx`);
    const model = onnxGanModels[modelName];
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

    const res: IOnnxGanRequest["res"] = {
        tensor: denormalized,
        dims: results.output.dims,
        inShape,
        outShape,
        imgSize: onnxGanModels[modelName].imgSize
    };

    return res;
}

export async function evalOnnxModelCGAN(modelName: TOnnxCganNames, batchSize: number, label: number){
    const file = path.join(rootDir, "assets", `${modelName}.onnx`);
    const model = onnxCganModels[modelName];
    const session = await InferenceSession.create(file);

    const randn = gaussian(0, 1);
    const x1 = new Float32Array(batchSize * model.inShape[1] * model.inShape[2] * model.inShape[3]);
    for(let i=0; i<x1.length; i++){
        x1[i] = randn.ppf(Math.random());
    }

    const inShape = [batchSize, model.inShape[1], model.inShape[2], model.inShape[3]];
    const noise = new Tensor("float32", x1, inShape);

    const x2 = new Int32Array(batchSize);
    const maxLabel = onnxCganModels[modelName].numClasses;
    for (let i=0; i<x2.length; i++){
        x2[i] = (label < 0 || label >= maxLabel)
            ? i%maxLabel
            : Math.ceil(label);
    }
    const labels = new Tensor("int32", x2, [batchSize]);
    const results = await session.run({ noise, labels });
    const outShape = [batchSize, model.outShape[1], model.outShape[2], model.outShape[3]];

    const denormalized: any[] = [];
    for(let i=0; i<results.output.data.length; i++){
        const value = results.output.data[i] as number;
        denormalized[i] = Math.floor((value + 1) * 127.5);
    }

    const res: IOnnxCganRequest["res"] = {
        tensor: denormalized,
        dims: results.output.dims,
        imgSize: onnxCganModels[modelName].imgSize,
        inShape,
        outShape,
        numClasses: onnxCganModels[modelName].numClasses,
        classes: onnxCganModels[modelName].classes
    };

    return res;
}