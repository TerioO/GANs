<script setup lang="ts">
import AppHeader from "./AppHeader.vue";
import AppFooter from "./AppFooter.vue";
import { onMounted } from "vue";
import { getServerStatus } from "../store/api";
import { useStore } from "../store/store";
import { useRoute, useRouter } from "vue-router";
import hljs from "highlight.js";

hljs.highlightAll();

const router = useRouter();
const route = useRoute();
const { setServerStatus } = useStore();
const { trigger, req } = getServerStatus();

onMounted(() => {
  const timeout = setTimeout(() => {
    setServerStatus("OFF");
    router.push("/waiting");
  }, 3000);
  trigger().then(() => {
    if (req.axiosRes?.ok) {
      clearTimeout(timeout);
      setServerStatus("ON");
      if (route.path === "/waiting") router.push("/");
    }
  });
});
</script>
<template>
  <div class="flex flex-col min-h-screen">
    <AppHeader />
    <div class="flex flex-1">
      <RouterView />
    </div>
    <AppFooter class="mt-auto" />
  </div>
</template>
