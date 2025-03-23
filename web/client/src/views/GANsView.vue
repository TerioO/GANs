<script setup lang="ts">
import { useStore } from "../store/store";

const { appHeaderRef } = useStore();

const activeClass = "font-bold  text-emerald-500";
</script>
<template>
  <main class="flex-1">
    <!-- Header -->
    <div
      class="px-4 pt-8 pb-4 sticky z-5 bg-white"
      :style="{ top: `${appHeaderRef?.clientHeight}px` }"
    >
      <div class="container mx-auto flex justify-center gap-4">
        <RouterLink to="/gans/linear" :active-class="activeClass">Linear</RouterLink>
        <RouterLink to="/" :active-class="activeClass">DCGAN</RouterLink>
      </div>
    </div>
    <RouterView />
  </main>
</template>
