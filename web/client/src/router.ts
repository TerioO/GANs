import { createRouter, createWebHistory } from "vue-router";
import { useStore } from "./store/store";
import HomeView from "./views/HomeView.vue";
import _404View from "./views/_404View.vue";
import GANsView from "./views/GANsView.vue";
import WaitServerView from "./views/WaitServerView.vue";

export const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: "/", name: "Home", component: HomeView },
    {
      path: "/gans",
      name: "GANs",
      component: GANsView,
      children: [
        {
          path: "linear",
          name: "Linear",
          component: () => import("./views/GANSimpleView.vue")
        },
        {
          path: "dcgan",
          name: "DCGAN",
          component: () => import("./views/DCGANView.vue")
        },
        {
          path: "cgan",
          name: "CGAN",
          component: () => import("./views/CGANView.vue")
        }
      ]
    },
    {
      path: "/waiting",
      name: "Waiting",
      component: WaitServerView
    },
    { path: "/:pathMatch(.*)*", name: "404 - Not Found", component: _404View }
  ]
});

router.beforeEach((to, _from, next) => {
  const store = useStore();

  if (store.serverStatus === "OFF" && to.name !== "Waiting") {
    return next({ name: "Waiting" });
  } else if (store.serverStatus === "ON" && to.name === "Waiting")
    return next({ name: "Home" });
  else next();
});
