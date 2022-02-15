<template>
  <div class="fixed-plugin" v-click-outside="closeDropDown">
    <div class="dropdown show-dropdown" :class="{ show: isOpen }" >
      <a data-toggle="dropdown">
        <i :class="`${selectedTools[currentToolId].iconName}`" style="color:#dfdfdf !important;padding-top:8px;padding-bottom:10px" @click="toggleDropDown"> </i>
      </a>
      <ul class="dropdown-menu" :class="{ show: isOpen }">
        <li class="header-title">Drawing Tools</li>
        <li class="adjustments-line text-center" style="height:50px !important;margin-top:0px">
          <span
            v-for="item in selectedTools"
            :key="item.type"
            class="badge filter"
            style="width:40px;height:40px;"
            :class="[`badge-${item.color}`, { active: item.active }]"
            v-on:click="changeDrawingTools(item.type)"
          >
          <i :class="`${item.iconName}`" style="margin-top:3px;" ></i>
          </span>
        </li>
        
        <li class="header-title">Stroke Width ({{strokeWidth}})</li>
        <li class="adjustments-line text-center" style="margin-bottom:10px">
          <vue-slider
            ref="slider"
            :min="1"
            :max="10"
            :interval="1"
            :marks="true"
            v-model="strokeWidth"
            v-bind="sliderOptions"
            style="width:80%;margin-left:25px;margin-top:0px;"
          ></vue-slider>
        </li>

        
      </ul>
    </div>
  </div>
</template>
<script>
import Vue from "vue";
import vueSlider from 'vue-slider-component';
import 'vue-slider-component/theme/antd.css'

import SocialSharing from "vue-social-sharing";
import VueGitHubButtons from "vue-github-buttons";
import "vue-github-buttons/dist/vue-github-buttons.css";
import { bus } from '../../../../bus'
Vue.use(SocialSharing);
Vue.use(VueGitHubButtons, { useCache: true });
export default {
  components:{
    vueSlider
  },
  data() {
    return {
      isOpen: false,
      strokeWidth:4,
      currentToolId:0,
      selectedTools:[
        { type:"tool-arrow", active: true, iconName:"fa fa-mouse-pointer fa-2x"},
        { type:"tool-brush", active: false, iconName:"fa fa-paint-brush fa-2x"},
        { type:"tool-eraser", active: false, iconName:"fa fa-eraser fa-2x"},
      ],
      sliderOptions:{
        stepStyle: void 1,
      }
    };
  },
  watch:{
    strokeWidth:function(val, oldVal){
      bus.$data.current_stroke_width = val
      bus.$emit("update:strokeWidth",val)    
    }
  },
  methods: {
    toggleDropDown() {
      this.isOpen = !this.isOpen;
      
      /*
      let active_item_id = -1;
      
      for (let i=0;i<this.selectedTools.length;i++){
        if (this.selectedTools[i].active == true){
          active_item_id = i
          this.selectedTools[i].active = false
        }
      }
      console.log(active_item_id)
      this.selectedTools[active_item_id].active = true
      */
    },
    closeDropDown() {
      this.isOpen = false;
    },
    toggleList(list, itemToActivate) {
      //list.forEach(listItem => {
      //  listItem.active = false;
      //});
      //itemToActivate.active = true;
    },
    updateValue(name, val) {
      console.log(name);
      this.$emit(`update:${name}`, val);
    },
    changeSidebarBackground(item) {
      this.$emit("update:color", item.color);
      this.toggleList(this.sidebarColors, item);
    },
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
  },
  mounted:function(){
    //bus.$emit('hack:updatebrush','tool-arrow')
    bus.$on('hack:updatebrush',()=>{
      this.$data.currentToolId = 0
      this.selectedTools[0].active = true
      this.selectedTools[1].active = false
      this.selectedTools[2].active = false
      bus.$data.current_drawing_tools= 'tool-arrow'
      this.$forceUpdate()
    })
  }
};
</script>
<style>
.centered-row {
  display: flex;
  height: 100%;
  align-items: center;
}

.button-container .btn {
  margin-right: 10px;
}

.centered-buttons {
  display: flex;
  justify-content: center;
}
</style>
