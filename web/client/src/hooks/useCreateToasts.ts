import { useToast } from "primevue";

export function useCreateToasts() {
  const toast = useToast();

  function displayApiError(message: string | null) {
    if(!message) return;
    toast.add({
      severity: "error",
      summary: "API Error",
      detail: message,
      life: 3000
    });
  }

  return {
    displayApiError
  };
}
