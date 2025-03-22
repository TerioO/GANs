<script setup lang="ts">
import { onMounted, useTemplateRef } from 'vue';
import { useApiStore } from '../store/apiStore';

const canvas = useTemplateRef("canvas");

const { getServerStatus, getGanSimpleV4 } = useApiStore();

function temp() {
    getServerStatus.trigger()
}

function getGan() {
    getGanSimpleV4.trigger({ batchSize: "15" }).then(() => {
        if (!canvas.value) return;
        const ctx = canvas.value.getContext("2d") as CanvasRenderingContext2D;
        if (!getGanSimpleV4.data) return;
        const { tensor, dims } = getGanSimpleV4.data;

        console.log(dims)

        const imgSize = 28;
        const batchSize = dims[0];
        const gridWidth = Math.ceil(Math.sqrt(batchSize));
        const gridHeight = Math.ceil(batchSize / gridWidth);

        const canvasW = imgSize * gridWidth;
        const canvasH = imgSize * gridHeight;
        canvas.value.width = canvasW;
        canvas.value.height = canvasW;
        const imageData = ctx.createImageData(canvasW, canvasH);

        for (let k = 0; k < batchSize; k++) {
            const gridX = (k % gridWidth) * imgSize;
            const gridY = Math.floor(k / gridWidth) * imgSize;
            for (let i = 0; i < imgSize; i++) {
                for (let j = 0; j < imgSize; j++) {
                    const tensorIdx = k * imgSize * imgSize + i * imgSize + j;
                    const value = tensor[tensorIdx]
                    const imgDataIdx = ((gridY + i) * canvasW + (gridX + j)) * 4;
                    imageData.data[imgDataIdx] = value;
                    imageData.data[imgDataIdx + 1] = value;
                    imageData.data[imgDataIdx + 2] = value;
                    imageData.data[imgDataIdx + 3] = 255;
                }
            }
        }
        ctx.putImageData(imageData, 0, 0);
    })
}

onMounted(() => {

})
</script>
<template>
    <main class="container mx-auto">
        <h1>HOME</h1>
        <button @click="temp">Get server status</button>
        <button @click="getGan">Get GAN images</button>
        <canvas ref="canvas" class="border-1 border-blue-500"></canvas>
        <div class="border-1 border-red-500 w-[28px] h-[28px]" ></div>
    </main>
</template>
