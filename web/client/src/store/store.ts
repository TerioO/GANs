import { defineStore } from "pinia";
import { ref } from "vue";

export const useStore = defineStore("store", () => {
  const appHeaderRef = ref<HTMLHeadElement | null>(null);
  const serverStatus = ref<"ON" | "OFF">("ON");

  function setAppHeaderRef(value: HTMLHeadElement) {
    appHeaderRef.value = value;
  }

  function setServerStatus(value: "ON" | "OFF") {
    serverStatus.value = value;
  }

  return {
    appHeaderRef,
    setAppHeaderRef,
    serverStatus,
    setServerStatus
  };
});
