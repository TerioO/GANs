import { defineStore } from "pinia";
import { ref } from "vue";

export const useStore = defineStore("store", () => {
    const appHeaderRef = ref<HTMLHeadElement | null>(null);
    const serverStatus = ref<"ON" | "OFF">("OFF");

    function setAppHeaderRef(value: HTMLHeadElement) {
        appHeaderRef.value = value;
    }

    function setServerStatus(value: "ON" | "OFF"){
        console.log(value)
        serverStatus.value = value;
        console.log(serverStatus.value)
    }

    return {
        appHeaderRef,
        setAppHeaderRef,
        serverStatus,
        setServerStatus
    }
})