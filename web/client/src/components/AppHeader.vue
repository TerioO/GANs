<script setup lang="ts">
import { useStore } from "../store/store";
import { onMounted, useTemplateRef } from "vue";

const header = useTemplateRef("header");

const { setAppHeaderRef } = useStore();

const activeClass = "text-emerald-500 font-bold";

onMounted(() => {
  if (!header.value) return;
  setAppHeaderRef(header.value);
});
</script>
<template>
  <header
    ref="header"
    class="p-4 sticky top-0 z-10 bg-white"
    style="box-shadow: 0px 0px 26px -14px black"
  >
    <div class="container mx-auto flex items-center gap-4">
      <RouterLink to="/" :active-class="activeClass">
        <i class="pi pi-home rounded-full !text-xl"></i>
      </RouterLink>
      <RouterLink to="/gans/linear" :active-class="activeClass"> GANs </RouterLink>
    </div>
  </header>
</template>
