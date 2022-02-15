<template>
  <div style="height:100%;overflow:hidden !important;background:#010101 !important; opacity:1 !important; border-right:1px #252525 solid; width:70px; position:absolute;">

    <div style="background-color:#010101 !important;top:margin-top:100px">
      <div style="height:70px;">
      </div>
      <div class="sidebar-tools" v-for="item in selectedTools" :key="item.type" 
        :class="[{ active: item.active }]"
        v-on:click="changeDrawingTools(item.type)"
      >
        <i :class="`${item.iconName}`" style="margin-top:20px;" ></i>
      </div>

      <div class="sidebar-tools" v-on:click="add_new_layer" style="margin-top:10px" >
        <i class="material-icons" style="margin-top:20px;">add</i>
      </div>

      <div class="sidebar-tools" v-on:click="clear_current_layer" >
        <i class="material-icons" style="margin-top:20px;">layers_clear</i>
      </div>

      <div class="sidebar-tools" v-on:click="askCopyCurrentLayer"  >
        <i class="material-icons" style="margin-top:20px;">content_paste</i>
      </div>
      
      <div class="sidebar-tools" v-on:click="delete_current_layer" >
        <i class="material-icons" style="margin-top:20px;">delete</i>
      </div>
      
      <div class="sidebar-tools" v-on:click="askAssembly" >
        <i class="material-icons" style="margin-top:20px;">dynamic_feed</i>
      </div>
      
      <div class="sidebar-tools" v-on:click="askChangeViewType" >
        <i class="material-icons"  style="margin-top:20px;">edit</i>
      </div>
      
      <div class="sidebar-tools" v-on:click="ask_load_image_to_current_layer" >
        <i class="material-icons"  style="margin-top:20px;">cloud_upload</i>
      </div>




    
    </div>

  </div>
</template>
<script>
import SidebarLink from "./SidebarLink.vue";
import {bus} from '../../../bus'
export default {
  components: {
    SidebarLink
  },
  data(){
    return{
      strokeWidth:4,
      currentToolId:0,
      selectedTools:[
        { type:"tool-arrow", active: true, iconName:"fa fa-mouse-pointer fa-2x"},
        { type:"tool-brush", active: false, iconName:"fa fa-paint-brush fa-2x"},
        { type:"tool-eraser", active: false, iconName:"fa fa-eraser fa-2x"},
      ],
    }
  },
  props: {
    title: {
      type: String,
      default: "Smart Sketch"
    },

  },
  provide() {
    return {
      autoClose: this.autoClose
    };
  },
  methods:{
    changeDrawingTools(item){
      let selectd_id = -1
      //console.log('drawing tool', item)
      for (let i=0;i<this.selectedTools.length;i++){
          if (this.selectedTools[i].type === item){
            this.selectedTools[i].active = true
            this.$data.currentToolId = i
          }else{
            this.selectedTools[i].active = false
          }
      }
      // change data then alter stage
      bus.$data.current_drawing_tools = item
      bus.$emit("update:drawingtools",item)
    },
    ask_load_image_to_current_layer(){
      console.log("ask load images")
      bus.$emit("load:images")
    },
    clear_current_layer(){
        //console.log("clear current layer")
        bus.$emit("update:clearChairLayer")
    },
    add_new_layer(){
        //console.log("clear current layer")
        bus.$emit("update:addChairLayer")
    },
    delete_current_layer(){
        bus.$emit("update:deleteChairLayer")
    },
    save_current_model(){
        bus.$emit("save:current:model")
    },    
    askCopyCurrentLayer(){
      bus.$emit("update:askPasteCurrentLayer")
    },
    askChangeViewType(){
      bus.$emit("update:askChangeViewType")
    },
    askAssembly(){
      bus.$emit("update:askAssembly")
    },
  },
  mounted:function(){
    bus.$on('hack:updatebrush',()=>{
      this.$data.currentToolId= 0
      this.$data.selectedTools[0].active = true
      this.$data.selectedTools[1].active = false
      this.$data.selectedTools[2].active = false
      bus.$data.current_drawing_tools = this.$data.selectedTools[0].type
      bus.$emit("update:drawingtools",this.$data.selectedTools[0].type)
    })
  }

};
</script>
<style lang="scss">
@media screen and (min-width: 991px) {
  .nav-mobile-menu {
    display: none;
  }
}
.sidebar-tools{
  height: 70px;
  text-align: center;
  color: lightgray;
  border: 1px solid rgb(37, 37, 37);
  &:hover {
    background-color: #116df7;
    transition-duration:.2s;
    transition-timing-function:cubic-bezier(.02,.01,.05,1);
  }

}
.active{
  background-color: rgb(17,109, 247);
}
.not-active{
  background-color: #010101;
}

.sidebar{
  background-color: #010101;
  opacity: 1;
}
</style>
