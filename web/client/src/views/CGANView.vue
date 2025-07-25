<script setup lang="ts">
import { watch, ref, onMounted } from "vue";
import { runCGAN } from "../store/api";
import { useCreateToasts } from "../hooks/useCreateToasts";
import {
  CODE_CDCGAN_Animal_Faces_v7,
  CODE_CDCGAN_Cats_v1,
  CODE_CDCGAN_MNIST_v3
} from "../assets/codeSnippets";
import type { Ref } from "vue";
import type { TOnnxCganNames } from "../types/api-types";
import OnnxCGANCanvas from "../components/OnnxCGANCanvas.vue";
import CodeModal from "../components/CodeModal.vue";

type TCGANS = {
  [key in TOnnxCganNames]: {
    req: ReturnType<typeof runCGAN>;
    data: Ref<OnnxCanvasData>;
  };
};

type OnnxCanvasData = {
  tensor: any[];
  dims: readonly number[];
  imgSize: number;
  channels: number;
} | null;

const { displayApiError } = useCreateToasts();

const CGANS: TCGANS = {
  CDCGAN_MNIST_v3: {
    req: runCGAN(),
    data: ref<OnnxCanvasData>(null)
  },
  CDCGAN_Cats_v1: {
    req: runCGAN(),
    data: ref<OnnxCanvasData>(null)
  },
  CDCGAN_Animal_Faces_v7: {
    req: runCGAN(),
    data: ref<OnnxCanvasData>(null)
  },
  CDCGAN_FashionMNIST_v2: {
    req: runCGAN(),
    data: ref<OnnxCanvasData>(null)
  }
};

onMounted(() => {
  Object.values(CGANS).forEach((el) => {
    watch(
      () => el.req.req.isErr,
      (val) => {
        if (val) displayApiError(el.req.req.errMsg);
      }
    );
  });
});

function generate(batchSize: number, label: number, modelName: TOnnxCganNames) {
  CGANS[modelName].req.trigger({ batchSize, label, modelName }).then((res) => {
    if (!res) return;
    CGANS[modelName].data.value = {
      tensor: res.data.tensor,
      dims: res.data.dims,
      imgSize: res.data.imgSize,
      channels: res.data.outShape[1]
    };
  });
}
</script>
<template>
  <section class="container mx-auto p-4 flex flex-col gap-4">
    <section class="mt-4">
      <p>
        <span class="font-bold">Label</span> input takes a number (int) represeting the
        class of the image you want to generate.
      </p>
      <p>
        Datasets can have multiple values for label, MNIST for example has 10 (0 to 9)
      </p>
      <p>
        If you insert a value bellow or above the dataset's existing classes, the API will
        return the labels in order (0 to n) if batch_size allows it
      </p>
    </section>
    <section class="mt-4">
      <h2 class="text-xl font-bold mb-4">CDCGAN_Animal_Faces_v7</h2>
      <div class="ml-4">
        <OnnxCGANCanvas
          :model-name="'CDCGAN_Animal_Faces_v7'"
          :loading="CGANS['CDCGAN_Animal_Faces_v7'].req.req.loading"
          :data="CGANS['CDCGAN_Animal_Faces_v7'].data.value"
          @generate="(b, l) => generate(b, l, 'CDCGAN_Animal_Faces_v7')"
        />
        <CodeModal
          class="mt-9"
          btn-label="View full code"
          header="CODE_CDCGAN_Animal_Faces_v7"
          :code="CODE_CDCGAN_Animal_Faces_v7"
        />
        <div class="flex flex-col mt-6">
          <p>Labels: [0, 2] (cat, dog, wild)</p>
          <p>Epochs: 680</p>
          <p>
            Dataset:
            <a
              href="https://www.kaggle.com/datasets/andrewmvd/animal-faces"
              target="_blank"
              class="text-blue-500 hover:underline break-all text-wrap"
              >https://www.kaggle.com/datasets/andrewmvd/animal-faces</a
            >
          </p>
        </div>
      </div>
    </section>
    <section class="mt-9">
      <h2 class="text-xl font-bold mb-4">CDCGAN_FashionMNIST_v2</h2>
      <div class="ml-4">
        <OnnxCGANCanvas
          :model-name="'CDCGAN_FashionMNIST_v2'"
          :loading="CGANS['CDCGAN_FashionMNIST_v2'].req.req.loading"
          :data="CGANS['CDCGAN_FashionMNIST_v2'].data.value"
          @generate="(b, l) => generate(b, l, 'CDCGAN_FashionMNIST_v2')"
        />
        <div class="flex flex-col mt-6">
          <p>
            Labels: [0, 10] ('T-shirt/top': 0, 'Trouser': 1, 'Pullover': 2, 'Dress': 3,
            'Coat': 4, 'Sandal': 5, 'Shirt': 6, 'Sneaker': 7, 'Bag': 8, 'Ankle boot': 9)
          </p>
          <p>Epochs: 100</p>
          <p>Dataset: PyTorch</p>
        </div>
      </div>
    </section>
    <section class="mt-9">
      <h2 class="text-xl font-bold mb-4">CDCGAN_MNIST_v3</h2>
      <div class="ml-4">
        <OnnxCGANCanvas
          :model-name="'CDCGAN_MNIST_v3'"
          :loading="CGANS['CDCGAN_MNIST_v3'].req.req.loading"
          :data="CGANS['CDCGAN_MNIST_v3'].data.value"
          @generate="(b, l) => generate(b, l, 'CDCGAN_MNIST_v3')"
        />
        <CodeModal
          class="mt-9"
          btn-label="View full code"
          header="CDCGAN_MNIST_v3"
          :code="CODE_CDCGAN_MNIST_v3"
        />
        <div class="flex flex-col mt-6">
          <p>Labels: [0, 10] (0, 1, 2, ..., 10)</p>
          <p>Epochs: 125</p>
          <p>Dataset: PyTorch</p>
        </div>
      </div>
    </section>
    <section class="mt-9">
      <h2 class="text-xl font-bold mb-4">CDCGAN_Cats_v1</h2>
      <div class="ml-4">
        <OnnxCGANCanvas
          :model-name="'CDCGAN_Cats_v1'"
          :loading="CGANS['CDCGAN_Cats_v1'].req.req.loading"
          :data="CGANS['CDCGAN_Cats_v1'].data.value"
          @generate="(b, l) => generate(b, l, 'CDCGAN_Cats_v1')"
        />
        <CodeModal
          class="mt-9"
          btn-label="View full code"
          header="CDCGAN_Cats_v1"
          :code="CODE_CDCGAN_Cats_v1"
        />
        <div class="flex flex-col mt-6">
          <p>Labels: [0, 1] (cat, dog)</p>
          <p>Epochs: 400</p>
          <p>
            Dataset:
            <a
              href="https://www.kaggle.com/datasets/tongpython/cat-and-dog"
              target="_blank"
              class="text-blue-500 hover:underline break-all text-wrap"
              >https://www.kaggle.com/datasets/tongpython/cat-and-dog</a
            >
          </p>
        </div>
      </div>
    </section>
  </section>
</template>
