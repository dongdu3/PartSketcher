import DashboardLayout from "@/pages/Layout/DashboardLayout.vue";

//import Dashboard from "@/pages/Dashboard.vue";
import Chair from "@/pages/Chair.vue";
import Bed from "@/pages/Bed.vue";
import Table from "@/pages/Table.vue";
const routes = [
  {
    path: "/",
    component: DashboardLayout,
    redirect: "/chair",
    children: [
      {
        path: "chair",
        name: "Chair Modeling",
        disable:false,
        component: Chair
      },{
        path: "lamp",
        name: "Lamp Modeling",
        disable:false,
        component: Chair
      },{
        path: "table",
        name: "Table Modeling",
        disable:false,
        component: Chair
      },   
    ]
  }
];

export default routes;
