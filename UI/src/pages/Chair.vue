<template>
  <div style="display:flex;padding-top:0px;margin-bottom:0px;height:100% !important; padding-bottom:0px;background:#121417;margin-left:70px;">
    <div class="canvascol" style="flex:1 1 auto">
        <div v-show="show_sketch" class="maincanvas" id="maincanvas" style="height:512px;width:512px;
            border-style:dashed;border-width:2px;border-color:#495057;">
        </div>
        <div v-show="show_preview" class="threecanvas" id="threecanvas" style="height:512px;width:512px;
              border-style:dashed;border-width:2px;border-color:#495057;">
        </div>

    </div>
    
    <div class="previecol" style="flex:0 0 340px;background-color:#010101;border-left:1px solid #252729;">
      <div style="height:200px;margin-top:10px;margin-left:30px;margin-right:30px;">
        
        <p style="font-size:16px; font-weight: bold;color:white;">Model Type</p>
          <el-dropdown>
            <el-button type="primary" 
              style="width:300px;margin-left:40px;margin-top:10px;
              background-color:rgb(17,109,247);border-width:0px;font-size:16px"
            >
              {{current_type}}  <i style="" class="el-icon-arrow-down el-icon--right"></i> 
            </el-button>
            <el-dropdown-menu slot="dropdown" style="width:300px;background-color:rgb(19,23,27)">
              <el-dropdown-item style="color:gray;" @click.native="change_current_type('Chair')" >Chair</el-dropdown-item>
              <el-dropdown-item style="color:gray;" @click.native="change_current_type('Lamp')">Lamp</el-dropdown-item>
              <el-dropdown-item style="color:gray;" @click.native="change_current_type('Table')">Table</el-dropdown-item>
            </el-dropdown-menu>
          </el-dropdown>
        
        <p style="font-size:16px; font-weight: bold;color:white; margin-top:30px;">Part Preview</p>
          
           <div id="previewlist" style="scroll-behavior: smooth;background-color:rgb(19,23,27);margin-right:0px; height:580px;max-height:580px;  overflow:scroll; border-radius: 5px; scrollbar-width:0px;">
            
              <md-card v-for="t in canvasLayers" v-bind:key="t.id" style="background-color:rgb(19,23,27);border-style:dashed;border-width:0px;border-color:#495057;width:382px; height:180px;margin:auto;margin-left:0px; margin-right:5px;margin-top:4px;margin-bottom:2px;">
                  <div style="height:170px;width:384px;">
                    <div style="margin-top:3px;margin-left:1px;width:30px;height:50px;display:inline-block;vertical-align:top">
                  
                      <i class="fa fa-eye" :class="t.visable?'layer-visable':'layer-notvisable'" style="margin-left:3px;padding-left:2px;
                      padding-right:2px;padding-top:2px;
                      height:22px;border-style:solid;border-width:2px;border-color:#495057;position:relative;"
                      v-on:click="change_visable_status(t.id)"
                      ></i>
                      <i class="fa fa-paint-brush" :class="t.activated?'layer-activated':'layer-notactivated'" style="margin-left:3px;margin-top:4px;padding-left:3px;
                      padding-right:3px;padding-top:2px;
                      height:22px;border-style:solid;border-width:2px;border-color:#495057;position:relative;"
                      v-on:click="change_activate_status(t.id)">
                      </i>
                    
                    </div>
                    
                    <div :id="t.previewId" style="background-color:white;border-radius:2px; vertical-align:top;top:0px;margin-top:3px;margin-left:0px;width:170px;height:170px;display:inline-block;position:relative;border-style:solid;border-width:1px;border-color:#495057;">
                    
                    </div>
                    <div :id="t.previewModelId" style="border-radius:2px; vertical-align:top;top:0px;margin-top:3px;margin-left:3px;width:170px;height:170px;display:inline-block;position:relative;border-style:solid;border-width:1px;border-color:#495057;">
                    
                    </div>
                  </div>
              
              </md-card>
          </div>
      </div>

    </div>
    <!-- 
    <div class="md-layout md-row">
      <div
        class="md-layout-item md-medium-size-50 md-xsmall-size-100 md-size-50 "
        style="margin-bottom:0px;min-height:400px;min-width:400px !important;max-width:480px;"
      >
      <md-card>
        <md-card-header style="background:#448aff;height:60px;">
          <h4 class="title" style="margin-top:-10px;">Sketch Pad View</h4>
          <p class="category" style="margin-top:-5px;" >Complete your Sketch Here</p>
        </md-card-header>
       
        <div class="maincanvas" id="maincanvas" style="height:334px;width:384px;
              margin:0 auto;margin-top:10px;margin-bottom:10px;border-style:dashed;border-width:2px;border-color:#495057;">
        </div>
        
        </md-card>

      </div>

      <div
        class="md-layout-item md-medium-size-50 md-xsmall-size-100 md-size-45" 
        style="margin-bottom:0px;min-height:400px;min-width:410px;min-height:760px;max-height:760px;"
      >
        <md-card>
          <md-card-header style="background:#448aff;height:60px;margin-bottom:10px;">
            <h4 class="title" style="margin-top:-10px;">Layer View</h4>
            <p class="category" style="margin-top:-5px;">Modify the Layers</p>
          </md-card-header>
          <md-content  style="max-height:760px; overflow-x:hidden; overflow-y:scroll;scrollbar-width:0px;">
            <md-card class=" md-elevation-5" v-for="t in canvasLayers" v-bind:key="t.id" style="border-style:dashed;border-width:2px;border-color:#495057;width:382px; height:180px;margin:auto;margin-left:15px; margin-top:4px;margin-bottom:2px;">
                <div style="height:170px;width:384px;">
                  <div style="margin-top:3px;margin-left:1px;width:30px;height:50px;display:inline-block;vertical-align:top">
                
                    <i class="fa fa-eye" :class="t.visable?'layer-visable':'layer-notvisable'" style="margin-left:3px;padding-left:2px;
                    padding-right:2px;padding-top:2px;
                    height:22px;border-style:solid;border-width:2px;border-color:#495057;position:relative;"
                    v-on:click="change_visable_status(t.id)"
                    ></i>
                    <i class="fa fa-paint-brush" :class="t.activated?'layer-activated':'layer-notactivated'" style="margin-left:3px;margin-top:4px;padding-left:3px;
                    padding-right:3px;padding-top:2px;
                    height:22px;border-style:solid;border-width:2px;border-color:#495057;position:relative;"
                    v-on:click="change_activate_status(t.id)">
                    </i>
                  
                  </div>
                   
                  <div :id="t.previewId" style="vertical-align:top;top:0px;margin-top:3px;margin-left:0px;width:170px;height:170px;display:inline-block;position:relative;border-style:solid;border-width:1px;border-color:#495057;">
                   
                  </div>
                  <div :id="t.previewModelId" style="vertical-align:top;top:0px;margin-top:3px;margin-left:3px;width:170px;height:170px;display:inline-block;position:relative;border-style:solid;border-width:1px;border-color:#495057;">
                   
                  </div>

              </div>



            </md-card>
          </md-content>
        
        </md-card>
      </div>
      <div
        class="md-layout-item md-medium-size-50 md-xsmall-size-100 md-size-50"
        style="margin-bottom:0px;min-height:300px;min-width:320px;max-width:480px;margin-top:-340px;"
      >
      <md-card>
        <md-card-header :style="this.assemble_head_color_type">
          <h4 class="title" style="margin-top:-10px;" >Assembly View</h4>
          <p class="category"  style="margin-top:-5px;">{{this.assemble_head_text}} </p>
        </md-card-header>
        
        <div class="threecanvas" id="threecanvas" style="height:330px;width:384px;
              margin:0 auto;margin-top:10px;margin-bottom:10px;border-style:dashed;border-width:2px;border-color:#495057;">
        </div>


        </md-card>
      </div>
    </div>
    -->
    
      


      <my-upload field="img"
            @crop-success="cropSuccess"
            @crop-upload-success="cropUploadSuccess"
            @crop-upload-fail="cropUploadFail"
            v-model="show_image_uploader"
        :width="512"
        :height="512"
        :params="params"
        :headers="headers"
        img-format="png"></my-upload>
      


      
  </div>
</template>

<script>

import {bus} from '../../bus'
import Vue from 'vue'; 
import _ from 'lodash'
import 'babel-polyfill'; // es6 shim
import myUpload from 'vue-image-crop-upload';

import { ModelObj } from 'vue-3d-model';
import * as Three from 'three'
const PLYLoader = require("threejs-ply-loader")(Three)
const plyLoader = new PLYLoader()

var OrbitControls = require('three-orbit-controls')(Three)
var TransformControls = require('three-transform-controls')(Three);



import {PLYExporter} from 'threejs-ext';
import dropdown from 'vue-dropdowns';

const cur_exporter = new PLYExporter()
//console.log(cur_exporter)

window.Three = Three;


export default {
  el: '#app',
  components: {
    ModelObj,
    'dropdown': dropdown,
    'my-upload': myUpload
  },

  data() {
    return {
      //-1 not assembled 0 assembled 1 changing pose of assembled
      view_mode:-1,
      cur_brush_width:5,
      cur_brush_type:'tool-arrow',
      current_type:'Chair',
      show_image_uploader:false,
      imgDataUrl:'',
      mesh_color:0x14A0C8,
      mesh_spec:0x111111,
      params: {
				token: '123456798',
				name: 'avatar'
			},
			headers: {
				smail: '*_~'
			},
      canvasLayers:[{

        id:0,
        items:[],
        activated:true,
        layer:-1,
        transHandle:-1,
        visable:true,
        previewStage:-1,
        previewLayer:-1,
        previewId:'preview-0',
        previewModelId:'preview-model-0',
        needsUpdate:false,
        current_3d_info:{
          cur_renderer:-1,
          cur_scene:-1,
          cur_mesh:-1,
          cur_scene_camera:-1,
          cur_scene_control:-1,
          cur_vox_info:-1,
        }
      }],
      show_sketch:true,
      show_preview:false,
      assemble_head_text:'Show The Assembled Model',
      assemble_head_color_type:'background:#448aff;height:60px;',
      download_link:-1,
      assemble_transform_controls:-1,
      assemble_mouse:-1,
      assemble_recaster:-1,
      selected_part_to_transform:null, 
      
      currentLayerId:0,
      canvas_height:512,
      canvas_width:512,
      preview_height:170,
      preview_width:170,
      global_stage:-1,
      is_painting:false,
      biggest_id:0,
      global_3d_info:{
        cur_renderer:-1,
        cur_scene:-1,
        cur_mesh:-1,
        cur_scene_camera:-1,
        cur_scene_control:-1,
        cur_mesh_string:-1,
      },
      assemble_mesh_part_list:[],
      INTERSECTED:null,
      layer_to_copy_id:null, 
      current_fps:30,
      current_time:0,
      three_clock:-1,


    };
  },
  methods:{
    
    load_image_to_current_layer(){
      let vid = -1
      for(let i = 0; i < this.$data.canvasLayers.length; i++){
        if (this.$data.canvasLayers[i].activated == true ){
          vid = i
        }
      }
      if(!(vid==-1)){
        if (!(this.$data.imgDataUrl=='')){
          var current_dong_image = new Konva.Image({
            width:512,
            height:512,
            draggable:false,
          })
          var dong_base_64_image = new Image()
          dong_base_64_image.src = this.$data.imgDataUrl
          
          current_dong_image.image(dong_base_64_image)
          this.$data.canvasLayers[vid].items.push(current_dong_image)
          this.$data.canvasLayers[vid].layer.add(this.$data.canvasLayers[vid].items[this.$data.canvasLayers[vid].items.length-1])
          this.$data.canvasLayers[vid].needsUpdate = true

          var debounced_update = _.debounce(this.update_part_sketch_models,200)

          this.redraw_visable_layers()
          this.redraw_preview_layers()
          debounced_update()
          
        }
      }
    },
    cropSuccess(imgDataUrl, field){
        
      console.log('-------- crop success --------');
      this.$data.imgDataUrl = imgDataUrl
      this.load_image_to_current_layer()
      
    },

		cropUploadSuccess(jsonData, field){
			//console.log('-------- upload success --------');
			//console.log(jsonData);
			//console.log('field: ' + field);
		},

		cropUploadFail(status, field){
			//console.log('-------- upload fail --------');
			//console.log(status);
			//console.log('field: ' + field);
		},
    change_current_type(cur_type_str){

      if (! (this.$data.current_type == cur_type_str)){
       
        this.$axios({
          method: "post",
          baseURL:'http://localhost:11451',
          url:'/changeModelType',
          crossDomain: true,
          header:{
            'Access-Control-Allow-Origin':true
          },
          data: {
            'modelType':cur_type_str
          }
        })
        .then(response => {
            this.$data.current_type = cur_type_str
        })
        .catch(error => console.log(error, "error"))
      }
    },
    synBusProps(){
      this.$data.cur_brush_width = bus.$data.current_stroke_width    
      this.$data.cur_brush_type = bus.$data.current_drawing_tools
    },
    change_activate_status(t_id){
      let vid = -1
      for(let i = 0;i< this.$data.canvasLayers.length; i++){
        if (t_id == this.$data.canvasLayers[i].id){
          vid = i
        }
      }
      
      if (!(vid == -1)){
        for(let i = 0; i< this.$data.canvasLayers.length;i++){
          if(i == vid){
            this.$data.canvasLayers[i].activated = true
          }else{
            this.$data.canvasLayers[i].activated = false
          }
        }
      }
      this.clearTransHandles()
      this.updateTransHandles()
      this.redraw_visable_layers()
      
    },
    change_visable_status(t_id){
      let vid = -1
      for(let i = 0;i< this.$data.canvasLayers.length; i++){
        if (t_id == this.$data.canvasLayers[i].id){
          vid = i
        }
      }
      if (!(vid == -1)){
        this.$data.canvasLayers[vid].visable = !this.$data.canvasLayers[vid].visable
        if (this.$data.canvasLayers[vid].visable == true){
          this.$data.canvasLayers[vid].layer.show()
        }else{
          this.$data.canvasLayers[vid].layer.hide()
        }

      }

      this.clearTransHandles()
      this.updateTransHandles()
      this.redraw_visable_layers()
    },
    change_type_option(cur_selected){
      //console.log('current selected',cur_selected)
      this.$data.selected_object = cur_selected
    },
    clear_canvas_dom(){
      var node= document.getElementById("maincanvas");

      this.$data.global_stage = -1

      if (node.querySelectorAll('*').length > 0){
        node.querySelectorAll('*').forEach(n => n.remove());
      }
    },
    change_canvas_icon_type(){
      if (this.$data.global_stage != -1){
        //this.$data.global_stage.container().style.cursor=
      }
    },
    
    redraw_visable_layers(){
      //console.log('redrawing')
      //this.$data.global_stage.clear
      for (let i = 0; i < this.$data.canvasLayers.length;i++){
        if (!this.$data.canvasLayers[i].visable==true){
            continue
        }
        if (!this.$data.canvasLayers[i].activated){
          //this.$data.canvasLayers[i].layer.opacity(1.0)
          //let allChildren = this.$data.canvasLayers[i].layer.getChildren()
          for(let t of this.$data.canvasLayers[i].layer.getChildren()){
            //console.log(t.globalCompositeOperation())
            if (t.globalCompositeOperation() == 'source-over'){
              t.opacity(0.3)
            }
          }
          //console.log('all children',allChildren)
        }else{
            for(let t of this.$data.canvasLayers[i].layer.getChildren()){
              t.opacity(1.0)
            }    
        }
        this.$data.canvasLayers[i].layer.batchDraw();

      }
    },

    clear_current_layer(){
      if (this.$data.show_preview){
        return
      }
      //find current layer id
      let vid = -1
      for(let i = 0; i < this.$data.canvasLayers.length; i++){
        if (this.$data.canvasLayers[i].activated == true ){
          vid = i
        }
      }
      //console.log('clear current layer id',vid)
      if (!(vid == -1)){
        //console.log('clear current layer',this.$data.canvasLayers[vid].layer)
        this.$data.canvasLayers[vid].items = []
        this.$data.canvasLayers[vid].layer.destroyChildren()
        this.$data.canvasLayers[vid].transHandle = -1
        this.$data.canvasLayers[vid].needsUpdate = false
        
        this.$data.canvasLayers[vid].current_3d_info.cur_mesh = -1
        
        var directionalLight = new Three.DirectionalLight( 0xffffff,1.35 );
        directionalLight.position.set(1, 1, 1);
        this.$data.canvasLayers[vid].current_3d_info.cur_scene = new Three.Scene()
        this.$data.canvasLayers[vid].current_3d_info.cur_scene.add(directionalLight)
        this.$data.canvasLayers[vid].current_3d_info.cur_scene.add(new Three.HemisphereLight( 0x443333, 0x111122 ))
        this.$data.canvasLayers[vid].transHandle = new Konva.Transformer({
          keepRatio: true,
          enabledAnchors: [
            'top-left',
            'top-right',
            'bottom-left',
            'bottom-right',
          ],
        })
        this.$data.canvasLayers[vid].layer.add(this.$data.canvasLayers[vid].transHandle)
      }
    },
    
    check_preview(){
      for (let i = 0; i < this.$data.canvasLayers.length; i++ ){
        if(this.$data.canvasLayers[i].previewStage == -1){
            //console.log('check preview',i)

        
            this.$data.canvasLayers[i].previewStage = new Konva.Stage({
            container: 'preview-' +  this.$data.canvasLayers[i].id,
            width: this.$data.preview_width,
            height: this.$data.preview_height,
            scaleX: this.$data.preview_width/this.$data.canvas_width,
            scaleY: this.$data.preview_height/this.$data.canvas_height
          });

        }
      }
    },
    check_model_preview(){
      
      for (let i = 0; i < this.$data.canvasLayers.length; i++){
        if(this.$data.canvasLayers[i].current_3d_info.cur_renderer == -1){
          
          let container = document.getElementById('preview-model-'+this.$data.canvasLayers[i].id)
          //console.log( container.clientWidth,' ', container.clientHeight)
          this.$data.canvasLayers[i].current_3d_info.cur_scene_camera = new Three.PerspectiveCamera(70, container.clientWidth / container.clientHeight, 0.01, 10)
          this.$data.canvasLayers[i].current_3d_info.cur_scene_camera.position.z = 1
          
          //this.$data.canvasLayers[i].current_3d_info.cur_mesh = new Three.Mesh(new Three.BoxGeometry(0.4, 0.4, 0.4), new Three.MeshNormalMaterial())
          this.$data.canvasLayers[i].current_3d_info.cur_scene = new Three.Scene()
          
          //this.$data.canvasLayers[i].current_3d_info.cur_scene.add(this.$data.canvasLayers[i].current_3d_info.cur_mesh)

          this.$data.canvasLayers[i].current_3d_info.cur_renderer = new Three.WebGLRenderer({ antialias: true })
          this.$data.canvasLayers[i].current_3d_info.cur_renderer.setSize(container.clientWidth, container.clientHeight)
          
          this.$data.canvasLayers[i].current_3d_info.cur_renderer.setClearColor(0xffffff,1);

          this.$data.canvasLayers[i].current_3d_info.cur_scene_control = new OrbitControls(this.$data.canvasLayers[i].current_3d_info.cur_scene_camera, this.$data.canvasLayers[i].current_3d_info.cur_renderer.domElement)
          this.$data.canvasLayers[i].current_3d_info.cur_scene_control.rotateSpeed = 1.0
          this.$data.canvasLayers[i].current_3d_info.cur_scene_control.zoomSpeed = 1.2
          this.$data.canvasLayers[i].current_3d_info.cur_scene_control.panSpeed = 0.8
          this.$data.canvasLayers[i].current_3d_info.cur_scene_control.enabled = true
          this.$data.canvasLayers[i].current_3d_info.cur_scene_control.enableRotate = true
          this.$data.canvasLayers[i].current_3d_info.cur_scene_control.enableZoom = true    
          
          container.appendChild(this.$data.canvasLayers[i].current_3d_info.cur_renderer.domElement)
          
        }

      }

    },
    add_new_layer(){
      if (this.$data.show_preview){
        return
      }
      
      for (let i = 0; i < this.$data.canvasLayers.length; i++ ){
        this.$data.canvasLayers[i].activated = false
      }

      this.$data.canvasLayers.push({
        id:this.$data.biggest_id + 1,
        items:[],
        activated:true,
        layer:-1,
        visable:true,
        previewStage:-1,
        needsUpdate:false,
        previewModelId:'preview-model-'+(this.$data.biggest_id+1),
        previewId:'preview-'+(this.$data.biggest_id+1),
        current_3d_info:{
          cur_renderer:-1,
          cur_scene:-1,
          cur_mesh:-1,
          cur_scene_camera:-1,
          cur_scene_control:-1,
          cur_vox_info:-1,

          
        }
      })
      //each height = 170
      let each_height = 170
      let sum_height = 580
      let border_upper = 10
      

      //console.log('after force updata',document.getElementById("preview-1"))
      


      for(let i = 0;i<this.$data.canvasLayers.length-2;i++){
        this.$data.canvasLayers[i].activated = false
      }
            
      this.$data.canvasLayers[this.$data.canvasLayers.length-1].layer = new Konva.Layer({
        id:('chair_'+(this.$data.canvasLayers[this.$data.canvasLayers.length-1].id).toString())
      });

      this.$data.biggest_id = this.$data.biggest_id + 1
      
      this.$data.canvasLayers[this.$data.canvasLayers.length-1].transHandle = new Konva.Transformer({
        keepRatio: true,
        enabledAnchors: [
          'top-left',
          'top-right',
          'bottom-left',
          'bottom-right',
        ],
      })
      //this.$data.canvasLayers[this.$data.canvasLayers.length-1].layer.add(this.$data.canvasLayers[this.$data.canvasLayers.length-1].transHandle)
      
      this.$data.global_stage.add(this.$data.canvasLayers[this.$data.canvasLayers.length-1].layer)
      
      if(this.$data.canvasLayers.length>3){
          this.$nextTick(()=>{
          console.log('ask scroll top')
          var container = document.querySelector('#previewlist');
          
          let cur_scroll_top = container.scrollTop
          
          let delta_current_to_fit = this.$data.canvasLayers.length * each_height + border_upper - sum_height + each_height
          
          container.scrollTop = delta_current_to_fit


          //console.log(container.scrollTop,delta_current_to_fit)
        })
        /*
        const el = document.getElementById('previewlist')

        let cur_scroll_top = el.scrollTop
        let delta_current_to_fit = this.$data.canvasLayers.length * each_height + border_upper - sum_height - cur_scroll_top
        */
      }

    },

    remove_current_layer(){
      if ( this.$data.canvasLayers.length <= 1 ){
        return
      }
      if (this.$data.show_preview){
        return
      }
      
      //find current layer id
      let vid = -1
      for(let i = 0; i < this.$data.canvasLayers.length; i++){
        if (this.$data.canvasLayers[i].activated == true ){
          vid = i
        }
      }
      
      if (!(vid == -1)){
        //console.log('clear current layer',this.$data.canvasLayers[vid].layer)
        this.$data.canvasLayers[vid].items = []
        this.$data.canvasLayers[vid].layer.destroyChildren()
        
      }
      
      this.$data.global_stage.find('#chair_'+(this.$data.canvasLayers[vid].id).toString()).destroy()

      if ( vid >=1 ){
        this.$data.canvasLayers[vid-1].activated = true
      }else{
        this.$data.canvasLayers[vid+1].activated = true
      }

      this.$data.canvasLayers.splice(vid,1)

      //console.log(this.$data.canvasLayers)
    },
    clearTransHandles(){
      //console.log('before cleaning ',this.$data.global_stage.find('Transformer'))
      for(let i = 0; i<this.$data.canvasLayers.length;i++){
        this.$data.canvasLayers[i].transHandle.nodes([])
        //for(let j = 0;j<this.$data.canvasLayers[i].layer.find('Transformer').length;j++){
        //}
      }
      
      this.$data.global_stage.find('Transformer').remove()
      //console.log('before cleaning ',this.$data.global_stage.find('Transformer'))
    },

    updateTransHandles(){
      //console.log(this.$data.canvasLayers[0].items)
      //find current layer id
      let vid = -1
      for(let i = 0; i < this.$data.canvasLayers.length; i++){
        if (this.$data.canvasLayers[i].activated == true ){
          vid = i
        }
      }
      //console.log('clear current layer id',vid)
      if (!(vid == -1)){
        //console.log(this.$data.canvasLayers[vid].items)
      for(let i = 0; i < this.$data.canvasLayers.length; i++){
        for(let j = 0; j < this.$data.canvasLayers[i].items.length; j++){

          if (i==vid && this.$data.cur_brush_type =='tool-arrow'){
            this.$data.canvasLayers[i].items[j].draggable(true)
            this.$data.canvasLayers[i].items[j].listening(true)
          }else{
            this.$data.canvasLayers[i].items[j].draggable(false)
            this.$data.canvasLayers[i].items[j].listening(false)
          }

        }
      }
        this.$data.canvasLayers[vid].transHandle.nodes(this.$data.canvasLayers[vid].items)    
        this.$data.canvasLayers[vid].layer.add(this.$data.canvasLayers[vid].transHandle)  
      }
    },

    inner_add_new_layer(){
      
      this.add_new_layer()
      
      this.clearTransHandles()
      
      if (bus.current_drawing_tools=='tool-arrow'){
        this.updateTransHandles()
      } 
      this.$nextTick(()=>{
        //console.log('start run')
        this.check_preview()
        this.check_model_preview()    
        this.redraw_visable_layers()
      })
      /*
      setTimeout(()=> {
        //console.log('start run')
        this.check_preview()
        this.check_model_preview()    
        this.redraw_visable_layers()
      }, 100); 
      */
    },

    copy_sketch_current_layers(){
      //console.log('this',this.$data.layer_to_copy_id)
      if (this.$data.show_preview){
        return
      }
      if(!(this.$data.layer_to_copy_id==null)){
        let vid = -1
        for(let i = 0; i < this.$data.canvasLayers.length; i++){
          if (this.$data.canvasLayers[i].activated == true ){
            vid = i
          }
        }
        //console.log('vid')
        for(let i = 0;i<this.$data.canvasLayers[this.$data.layer_to_copy_id].items.length;i++){
          this.$data.canvasLayers[vid].items.push(this.$data.canvasLayers[this.$data.layer_to_copy_id].items[i].clone({
            x:10,
            y:10,
            draggable:false,
          }))
          this.$data.canvasLayers[vid].layer.add(this.$data.canvasLayers[vid].items[i])
        }
        this.$data.canvasLayers[vid].current_3d_info.cur_mesh = this.$data.canvasLayers[this.$data.layer_to_copy_id].current_3d_info.cur_mesh.clone()
        this.$data.canvasLayers[vid].current_3d_info.cur_scene.add(this.$data.canvasLayers[vid].current_3d_info.cur_mesh)
        var directionalLight = new Three.DirectionalLight( 0xffffff,1.35 );
          
        directionalLight.position.set(1, 1, 1);
        this.$data.canvasLayers[vid].current_3d_info.cur_scene.add(directionalLight)
        this.$data.canvasLayers[vid].current_3d_info.cur_scene.add(new Three.HemisphereLight( 0x443333, 0x111122 ))
        //this.$data.canvasLayers.current_3d_info.cur_vox_info
        this.$data.canvasLayers[vid].current_3d_info.cur_vox_info = this.$data.canvasLayers[this.$data.layer_to_copy_id].current_3d_info.cur_vox_info
        //this.$data.canvasLayers[vid].layer = this.$data.canvasLayers[this.$data.layer_to_copy_id].layer.clone()

        //this.$data.canvasLayers[i].previewLayer = this.$data.canvasLayers[i].layer.clone({ hitGraphEnabled: false })
        this.$data.layer_to_copy_id = null
        
      }
    },

    copy_current_layer(){
      console.log('copy current layer')
      let vid = -1
      for(let i = 0; i < this.$data.canvasLayers.length; i++){
        if (this.$data.canvasLayers[i].activated == true ){
          vid = i
        }
      }
      this.$data.layer_to_copy_id = vid
      //console.log('current vid',this.$data.layer_to_copy_id)

      //return 

      //load previous layer
      this.add_new_layer()
      
      this.clearTransHandles()
      
      if (bus.current_drawing_tools=='tool-arrow'){
        this.updateTransHandles()
      }
      this.$nextTick(()=>{
        this.check_preview()
        this.check_model_preview()
        
        this.copy_sketch_current_layers()
        this.redraw_visable_layers()
        this.update_part_sketch_models()
        this.redraw_preview_layers()
        bus.$emit('hack:updatebrush','tool-arrow')
        //bus.current_drawing_tools = 'tool-arrow'
        this.synBusProps()
        this.updateTransHandles()
      })
      /*
      setTimeout(()=> {
        //console.log('start run')
        this.check_preview()
        this.check_model_preview()
        
        this.copy_sketch_current_layers()
        this.redraw_visable_layers()
        this.update_part_sketch_models()
        this.redraw_preview_layers()
        bus.$emit('hack:updatebrush','tool-arrow')
        //bus.current_drawing_tools = 'tool-arrow'
        this.synBusProps()
        this.updateTransHandles()        
      }, 100); 
      */
    },
    
    inner_remove_layer(){
      
      this.remove_current_layer()
      
      this.clearTransHandles()
      if (bus.current_drawing_tools=='tool-arrow'){
        this.updateTransHandles()
      } 

      this.redraw_visable_layers()
    },

    
    redraw_preview_layers(){
      
      //console.log('redraw preview')
      for(let i = 0; i < this.$data.canvasLayers.length;i++){
        //clear the layers
        
        if (!this.$data.canvasLayers[i].previewLayer==-1){
          this.$data.canvasLayers[i].previewLayer.destroy()
        }
        this.$data.canvasLayers[i].previewStage.destroyChildren()
        this.$data.canvasLayers[i].previewLayer = this.$data.canvasLayers[i].layer.clone({ hitGraphEnabled: false })
        

        this.$data.canvasLayers[i].previewLayer.show()
        
        for(let t of this.$data.canvasLayers[i].previewLayer.getChildren()){
          t.opacity(1.0)
        }

        this.$data.canvasLayers[i].previewStage.add(this.$data.canvasLayers[i].previewLayer)
        this.$data.canvasLayers[i].previewLayer.batchDraw()

        //console.log(i,'th layer ',this.$data.canvasLayers[i].previewLayer)
        //update new layers
      }

    },

    redraw_global_scene(){
      requestAnimationFrame(this.redraw_global_scene)
      
      let T = this.$data.three_clock.getDelta()
      this.$data.current_time = this.$data.current_time + T
      let renderT =1.0/this.$data.current_fps
      
      if (this.$data.current_time >renderT ){

        this.$data.global_3d_info.cur_scene_control.update()
      
        for(let i = 0; i < this.$data.canvasLayers.length; i++ ){
          //console.log(i,' ', this.$data.canvasLayers[i].current_3d_info.cur_scene_control)
          if (!(this.$data.canvasLayers[i].current_3d_info.cur_scene_control == -1) ){
            
            this.$data.canvasLayers[i].current_3d_info.cur_scene_control.update()
            this.$data.canvasLayers[i].current_3d_info.cur_renderer.render(this.$data.canvasLayers[i].current_3d_info.cur_scene, this.$data.canvasLayers[i].current_3d_info.cur_scene_camera)
          
          }
        
        }
        if (this.$data.view_mode == 1){

          this.$data.assemble_recaster.setFromCamera(this.$data.assemble_mouse,this.$data.global_3d_info.cur_scene_camera)

          var intersects = this.$data.assemble_recaster.intersectObjects(this.$data.global_3d_info.cur_scene.children)
          if (intersects.length > 0){
            if (this.$data.INTERSECTED != intersects[0].object){
              if(this.$data.INTERSECTED){
                this.$data.INTERSECTED.material.emissive.setHex(this.$data.INTERSECTED.currentHex)
              }
              this.$data.INTERSECTED = intersects[0].object
              this.$data.INTERSECTED.currentHex = this.$data.INTERSECTED.material.emissive.getHex()
              this.$data.INTERSECTED.material.emissive.setHex( 0xff0000 );

            }
            //console.log('interect num',intersects.length)
          }else{
            if (this.$data.INTERSECTED){
              this.$data.INTERSECTED.material.emissive.setHex(this.$data.INTERSECTED.currentHex)
              this.$data.INTERSECTED = null
            }
          }
        }
        this.$data.global_3d_info.cur_renderer.render(this.$data.global_3d_info.cur_scene, this.$data.global_3d_info.cur_scene_camera)
        this.$data.current_time = 0
      }
    },
    assemble_parts(){
      let whole_image_url = this.$data.global_stage.toDataURL()
      let part_image_url = []
      let part_vox_arr = []
      //get whoel sketch image

      for(let i = 0;i<this.$data.canvasLayers.length;i++){
        part_image_url.push(this.$data.canvasLayers[i].previewStage.toDataURL())
        part_vox_arr.push(JSON.parse(this.$data.canvasLayers[i].current_3d_info.cur_vox_info))
      }   

      this.$axios({
        method: "post",
        baseURL:'http://localhost:11451',
        url:'/assembleFromImagesNew',
        crossDomain: true,
        header:{
          'Access-Control-Allow-Origin':true
        },
        data: {
          'whole_image':whole_image_url,
          'part_image':part_image_url,
          'part_vox':part_vox_arr
        }
      })
      .then(response => {
          let cur_container = document.getElementById('threecanvas')
          
          cur_container.removeEventListener("mousemove", this.onDocumentMouseMove);
          cur_container.removeEventListener('dblclick',this.onDoubleClickItem);
          document.removeEventListener('keydown',this.onChangeTransfromType,false)

          this.$data.global_3d_info.cur_mesh =  new Three.Mesh(plyLoader.parse(response.data['assembled_model']), new Three.MeshPhongMaterial( {  color: this.$data.mesh_color, flatShading: true ,specular: this.$data.mesh_spec, shininess: 100} ))
          this.$data.global_3d_info.cur_mesh_string = response.data['assembled_model']
          this.$data.assemble_transform_controls = -1
          this.$data.global_3d_info.cur_scene = new Three.Scene()
          var directionalLight = new Three.DirectionalLight( 0xffffff,1.35 );
          
          directionalLight.position.set(1, 1, 1);
          this.$data.global_3d_info.cur_scene.add(directionalLight)
          this.$data.global_3d_info.cur_scene.add(new Three.HemisphereLight( 0x443333, 0x111122 ))
          this.$data.global_3d_info.cur_scene.add(this.$data.global_3d_info.cur_mesh)
          //console.log('voxel mesh', this.$data.global_3d_info.cur_mesh);

          //console.log('response data', response.data)
          this.$data.assemble_mesh_part_list = []
          //for each model we store its info
          for (let i=0;i<response.data['each_part_mesh'].length;i++){
            this.$data.assemble_mesh_part_list.push({
              'part_mesh': (new Three.Mesh(plyLoader.parse(response.data['each_part_mesh'][i]), new Three.MeshPhongMaterial( {  color: this.$data.mesh_color, flatShading: true, specular: this.$data.mesh_spec, shininess: 100} ))),
              'part_mesh_string':response.data['each_part_mesh'][i],
            })
          }
          // renew the data to be assembled
          
          this.$data.view_mode = 0
          this.$data.assemble_head_color_type='background:#448aff;height:60px;'
          this.$data.assemble_head_text = 'Show The Assembled Model'
          if (this.$data.show_preview==false){
            this.$data.show_preview = true
            this.$data.show_sketch = false
            bus.$emit('hack:header')
          }
          //here we have to remove the 
      })
      .catch(error => console.log(error, "error"))
    },

    init_sketch_server(){
      this.$axios({
        method: "post",
        baseURL:'http://localhost:11451',
        url:'/initModel',
        crossDomain: true,
        header:{
          'Access-Control-Allow-Origin':true
        },
        data: {
          keyword: "1"   
        }
      })
      .then(response => {
          console.log(response, "success")
      })
      .catch(error => console.log(error, "error"))
    
    },

    update_part_sketch_models(){
        console.log('ask update sketch part models')
        let part_image_url = []
        //get whoel sketch image

        for(let i = 0;i<this.$data.canvasLayers.length;i++){
          part_image_url.push(this.$data.canvasLayers[i].previewStage.toDataURL())
        }
        
        this.$axios({
          method: "post",
          baseURL:'http://localhost:11451',
          url:'/inferAllParts',
          crossDomain: true,
          header:{
            'Access-Control-Allow-Origin':true
          },
          data: {
            'part_image':part_image_url
          }
        })
        .then(response => {
            //console.log("success",response,)
            //let cur_model = response.data['all_parts'][0]
            // clear seene
            for(let i  = 0; i< response.data['all_parts'].length;i++){
              
              if (!this.$data.canvasLayers[i].needsUpdate){
                continue
              }
              this.$data.canvasLayers[i].needsUpdate = false
              this.$data.canvasLayers[i].current_3d_info.cur_vox_info = JSON.stringify(response.data['all_voxes'][i])
            
              this.$data.canvasLayers[i].current_3d_info.cur_mesh = new Three.Mesh(plyLoader.parse(response.data['all_parts'][i]), new Three.MeshPhongMaterial( {  color: this.$data.mesh_color, flatShading: true ,specular: this.$data.mesh_spec, shininess: 100} ))            
              var directionalLight = new Three.DirectionalLight( 0xffffff,1.35 );
              directionalLight.position.set(1, 1, 1);
              this.$data.canvasLayers[i].current_3d_info.cur_scene = new Three.Scene()
              this.$data.canvasLayers[i].current_3d_info.cur_scene.add(directionalLight)
              this.$data.canvasLayers[i].current_3d_info.cur_scene.add(new Three.HemisphereLight( 0x443333, 0x111122 ))
              this.$data.canvasLayers[i].current_3d_info.cur_scene.add(this.$data.canvasLayers[i].current_3d_info.cur_mesh)  
            }

        })
        .catch(error => console.log(error, "error"))

    },
    onDoubleClickItem(){
      console.log('current selected part',this.$data.selected_part_to_transform)
      if (this.$data.INTERSECTED){
        if(this.$data.selected_part_to_transform != this.$data.INTERSECTED){
          this.$data.selected_part_to_transform = this.$data.INTERSECTED
          //clear the transform
          this.$data.assemble_transform_controls.detach()
          this.$data.assemble_transform_controls.attach(this.$data.selected_part_to_transform)
        }
      }
      
    },
    onSaveCurrentModel(){

      console.log('saveing current model')
      
      let result = this.$data.global_3d_info.cur_mesh_string

      //console.log('after save model', result)
      this.$data.download_link.href = URL.createObjectURL(  new Blob( [ result ], { type: 'text/plain' } ) );
      this.$data.download_link.download = 'results.ply'
      this.$data.download_link.click()
        
    },
    onAskChangeModelType(){

      let cur_model_URL_splitted = document.URL.split('/')
      let cur_model_type = cur_model_URL_splitted[cur_model_URL_splitted.length-1]
      cur_model_type = cur_model_type.charAt(0).toUpperCase() + cur_model_type.slice(1);

      console.log('current model type',cur_model_type)
      
      this.$axios({
          method: "post",
          baseURL:'http://localhost:11451',
          url:'/changeModelType',
          crossDomain: true,
          header:{
            'Access-Control-Allow-Origin':true
          },
          data: {
            'modelType':cur_model_type
          }
        })
        .then(response => {
        })
        .catch(error => console.log(error, "error"))

    
    },
    onChangeTransfromType(event){
      
      switch(event.keyCode){
        	case 87: // W
					this.$data.assemble_transform_controls.setMode( "translate" );
          break;
          case 82: // R
					this.$data.assemble_transform_controls.setMode( "scale" );
          break;
      }
    },
    onDocumentMouseMove(event){

      event.preventDefault()
      
      let container = document.getElementById('threecanvas')
      this.$data.assemble_mouse.x = (event.offsetX/container.clientWidth) *2 - 1
      this.$data.assemble_mouse.y = - (event.offsetY/container.clientHeight)*2 + 1
      //console.log(event.offsetX, event.offsetY)
      //console.log("assemble mouse positoin",this.$data.assemble_mouse.x,this.$data.assemble_mouse.y, event.clientX, event.clientY,container.clientWidth,container.clientHeight)
  
    },
    onChangeViewType(){
      if (!this.$data.show_preview){
        return
      }
      // check if it has been assembled
      // already assembly mode
      if (this.$data.view_mode == 0){
        // clear current scene
        // generate the scene to move the conmponets 
        // this.$data.global_3d_info.cur_mesh =  new Three.Mesh(plyLoader.parse(response.data['assembled_model']), new Three.MeshPhongMaterial( {  color: 0x0055ff, flatShading: true ,specular: 0x111111, shininess: 100} ))

        this.$data.global_3d_info.cur_scene = new Three.Scene()
        var directionalLight = new Three.DirectionalLight( 0xffffff,1.35 );
          
        directionalLight.position.set(1, 1, 1);
        this.$data.global_3d_info.cur_scene.add(directionalLight)
        this.$data.global_3d_info.cur_scene.add(new Three.HemisphereLight( 0x443333, 0x111122 ))
        this.$data.assemble_mouse = new Three.Vector2();
        this.$data.assemble_mouse.x = -10
        this.$data.assemble_mouse.y = -10
        
        this.$data.assemble_transform_controls = new TransformControls(this.$data.global_3d_info.cur_scene_camera,this.$data.global_3d_info.cur_renderer.domElement)
        this.$data.assemble_transform_controls.space ='local'
        this.$data.assemble_transform_controls.setSize(2)
        for(let i = 0;i<this.$data.assemble_mesh_part_list.length;i++){

            this.$data.global_3d_info.cur_scene.add(this.$data.assemble_mesh_part_list[i].part_mesh)
            
            //this.$data.assemble_transform_controls.attach(this.$data.assemble_mesh_part_list[i].part_mesh)
        }
        this.$data.global_3d_info.cur_scene.add(this.$data.assemble_transform_controls)
        //this.$data.assemble_transform_controls.addEventListener('change',this.$data.global_3d_info.cur_renderer.render(this.$data.global_3d_info.cur_scene, this.$data.global_3d_info.cur_scene_camera))
        this.$data.view_mode = 1
        this.$data.assemble_head_text = 'Edit the model'
        this.$data.assemble_head_color_type='background:#0d47a1;height:60px;'
        this.$data.assemble_recaster = new Three.Raycaster()
        console.log('change to mode',this.$data.view_mode)
        let cur_container = document.getElementById('threecanvas')
        cur_container.addEventListener('mousemove',this.onDocumentMouseMove,false)
        cur_container.addEventListener('dblclick',this.onDoubleClickItem,false)
        document.addEventListener('keydown',this.onChangeTransfromType,false)
        //cur_container.addEventListener('keydown',this.onChangeTransfromType,false)

      }
      else if(this.$data.view_mode == 1){     

        let transform_information_arr = []
        let scale_information_arr = []
        let mesh_string_information_arr =[]
        //let part_vox_information_arr = []
        for (let i = 0 ; i<this.$data.assemble_mesh_part_list.length;i++){
          scale_information_arr.push([
            this.$data.assemble_mesh_part_list[i].part_mesh.scale.x,
            this.$data.assemble_mesh_part_list[i].part_mesh.scale.y,
            this.$data.assemble_mesh_part_list[i].part_mesh.scale.z
          ])
          transform_information_arr.push([
            this.$data.assemble_mesh_part_list[i].part_mesh.position.x,
            this.$data.assemble_mesh_part_list[i].part_mesh.position.y,
            this.$data.assemble_mesh_part_list[i].part_mesh.position.z
          ])
          mesh_string_information_arr.push(
            this.$data.assemble_mesh_part_list[i].part_mesh_string
          )
          //part_vox_information_arr.push(
          //  this.$data.assemble_mesh_part_list[i].part_vox
          //)
        }
        
        for(let i = 0;i< this.$data.assemble_mesh_part_list.length;i++){
          console.log(i,' ', this.$data.assemble_mesh_part_list[i].part_mesh.scale,' ',this.$data.assemble_mesh_part_list[i].part_mesh.position)
        }

        this.$axios({
          method: "post",
          baseURL:'http://localhost:11451',
          url:'/generateTransformedResults',
          crossDomain: true,
          header:{
            'Access-Control-Allow-Origin':true
          },
          data: {
            'mesh_string_arr':mesh_string_information_arr,
            'transform_arr':transform_information_arr,
            'scale_arr':scale_information_arr,
            //'part_vox_arr':part_vox_information_arr,
          }
        })
        .then(response => {
          // send the pose to the back door
          // assemble again
          // for testing
          this.$data.view_mode = 0
          console.log('change to mode',this.$data.view_mode)
          this.$data.assemble_head_text = 'Show The Assembled Model'
          this.$data.assemble_head_color_type='background:#448aff;height:60px;'
          this.$forceUpdate()
          this.$data.assemble_transform_controls = -1
          let cur_container = document.getElementById('threecanvas')
          cur_container.removeEventListener('mousemove',this.onDocumentMouseMove)
          cur_container.removeEventListener('dblclick',this.onDoubleClickItem);
          document.removeEventListener('keydown',this.onChangeTransfromType,false)

          this.$data.global_3d_info.cur_mesh =  new Three.Mesh(plyLoader.parse(response.data['assembled_model']), new Three.MeshPhongMaterial( {   color: this.$data.mesh_color, flatShading: true ,specular: this.$data.mesh_spec, shininess: 100} ))
          this.$data.global_3d_info.cur_mesh_string = response.data['assembled_model']
          this.$data.assemble_transform_controls = -1
          this.$data.global_3d_info.cur_scene = new Three.Scene()
          var directionalLight = new Three.DirectionalLight( 0xffffff,1.35 );
          
          directionalLight.position.set(1, 1, 1);
          this.$data.global_3d_info.cur_scene.add(directionalLight)
          this.$data.global_3d_info.cur_scene.add(new Three.HemisphereLight( 0x443333, 0x111122 ))
          this.$data.global_3d_info.cur_scene.add(this.$data.global_3d_info.cur_mesh)
          //console.log('voxel mesh', this.$data.global_3d_info.cur_mesh);

          //console.log('response data', response.data)
          this.$data.assemble_mesh_part_list = []
          //for each model we store its info
          for (let i=0;i<response.data['each_part_mesh'].length;i++){
            this.$data.assemble_mesh_part_list.push({
              'part_mesh': (new Three.Mesh(plyLoader.parse(response.data['each_part_mesh'][i]), new Three.MeshPhongMaterial( {   color: this.$data.mesh_color, flatShading: true ,specular: 0x101010, shininess: 100} ))),
              'part_mesh_string':response.data['each_part_mesh'][i],
            })
          }
        })
        .catch(error => console.log(error, "error"))
      }
    }

  },

  mounted:function(){
    
    bus.$on("load:images",()=>{
      this.$data.show_image_uploader = true || this.$.data.show_image_uploader
    })

    bus.$on("update:changeviewtype",(cur_type)=>{
      
      let this_is_sketch = true
      if(cur_type == 'Sketch'){
        this_is_sketch = true
      }else{
        this_is_sketch = false
      }
      if (! (this_is_sketch == this.$data.show_sketch)){
        this.$data.show_sketch = this_is_sketch
        this.$data.show_preview = !this.$data.show_sketch
      }

    })
    
    this.$data.view_mode = -1
    this.clear_canvas_dom()
    this.init_sketch_server()
    var debouncedPasteCurrentLayer = _.debounce(this.copy_current_layer,1000)
    var debouncedChnageViewType = _.debounce(this.onChangeViewType,1000)
    var debouncedSaveModel = _.debounce(this.onSaveCurrentModel,1000)
    var debouncedChangeModelType = _.debounce(this.onAskChangeModelType,1000)

    //ask model to change
    this.onAskChangeModelType()

    bus.$on('update:askPasteCurrentLayer',()=>{
      debouncedPasteCurrentLayer()
    })

    bus.$on('update:askChangeViewType',()=>{
      debouncedChnageViewType()
    })

    bus.$on('update:drawingtools',(text)=>{
      
      this.synBusProps()
      this.change_canvas_icon_type()

      this.clearTransHandles()
      
      //20206012 here need something to make all the drawn not draggable

      if (bus.current_drawing_tools=='tool-arrow'){
        this.updateTransHandles()
      } 
    
    })
    
    bus.$on('update:strokeWidth',(text)=>{
      this.synBusProps()
    })
  
    bus.$on('update:modeltype',debouncedChangeModelType)

    bus.$on('update:askAssembly',()=>{
      console.log('ask assembly')
      this.assemble_parts()

    })

    bus.$on('save:current:model',()=>{
      debouncedSaveModel()
    })

    bus.$on('update:clearChairLayer',()=>{
      var debounced_update = _.debounce(this.update_part_sketch_models,100)
      this.clear_current_layer()

      this.clearTransHandles()
      
      if (bus.current_drawing_tools=='tool-arrow'){
        this.updateTransHandles()
      } 
      
      this.redraw_visable_layers()
      this.redraw_preview_layers()
    
      debounced_update()
      
    })
    
    bus.$on('update:addChairLayer',_.debounce(this.inner_add_new_layer,200))

    bus.$on('update:deleteChairLayer',_.debounce(this.inner_remove_layer,200))

    this.$data.global_stage = new Konva.Stage({
      container: 'maincanvas',
      width: this.$data.canvas_height,
      height: this.$data.canvas_width
    });

    
    
    this.$data.canvasLayers[0].layer = new Konva.Layer({
        id:('chair_'+(this.$data.canvasLayers[this.$data.canvasLayers.length-1].id).toString())
      }
    );
    
    
    this.$data.canvasLayers[0].previewStage = new Konva.Stage({
      container: 'preview-' +  this.$data.canvasLayers[0].id.toString(),
      width: this.$data.preview_width,
      height: this.$data.preview_height,
      scaleX: this.$data.preview_width/this.$data.canvas_width,
      scaleY: this.$data.preview_height/this.$data.canvas_height
    });

    this.$data.canvasLayers[0].transHandle = new Konva.Transformer({
        keepRatio: true,
        enabledAnchors: [
          'top-left',
          'top-right',
          'bottom-left',
          'bottom-right',
        ],
    })
    

    
    //this.$data.canvasLayers[0].layer.add(this.$data.canvasLayers[0].transHandle)


    this.$data.global_stage.add(this.$data.canvasLayers[0].layer)



    this.$data.global_stage.on('mousedown touchstart',(e)=>{
      
      this.$data.is_painting = true
      
      var pos = this.$data.global_stage.getPointerPosition()
      
      //console.log(this.$data.cur_brush_type)
      
      if (this.$data.cur_brush_type == 'tool-brush' || this.$data.cur_brush_type == 'tool-eraser' ){
        let tid = -1
        for (let i = 0; i < this.$data.canvasLayers.length; i++){
          if (this.$data.canvasLayers[i].activated == true){
            tid = i
          }
        }
        
        this.$data.canvasLayers[tid].needsUpdate = true

        //console.log('tid',tid,this.$data.canvasLayers[tid])
        
        this.$data.canvasLayers[tid].items.push(new Konva.Line({
          stroke: 'black',
          strokeWidth: this.$data.cur_brush_width,
          globalCompositeOperation: this.$data.cur_brush_type === 'tool-brush' ? 'source-over' : 'destination-out',
          points: [pos.x, pos.y],
          draggable:false,
          //listening:false
        }))
        
        //this.$data.canvasLayers[tid].transHandle.nodes(this.$data.canvasLayers[tid].items)
        this.$data.canvasLayers[tid].layer.add(this.$data.canvasLayers[tid].items[(this.$data.canvasLayers[tid].items.length-1)])
        }
      
      })

      this.$data.global_stage.on('mouseup touchend', () => {
        var debounced_update = _.debounce(this.update_part_sketch_models,200)
        this.$data.is_painting = false;
        this.redraw_preview_layers()
        debounced_update()
      });
    
      this.$data.global_stage.on('mousemove touchmove', () => {
        
        if (!this.$data.is_painting) {
          return;
        }
        if (!(this.$data.cur_brush_type == 'tool-brush' || this.$data.cur_brush_type == 'tool-eraser') ) {
          return;
        }
        const pos = this.$data.global_stage.getPointerPosition();
        
        let tid = -1
        for (let i = 0; i < this.$data.canvasLayers.length; i++){
          if (this.$data.canvasLayers[i].activated == true){
            tid = i
          }
        }

      var newPoints = this.$data.canvasLayers[tid].items[(this.$data.canvasLayers[tid].items.length-1)].points().concat([pos.x, pos.y]);


      this.$data.canvasLayers[tid].items[(this.$data.canvasLayers[tid].items.length-1)].points(newPoints);
        //layer.batchDraw();
      this.redraw_visable_layers()
      
    });
    
    let hacked_width = 512
    let hacked_height = 512

    //start init 3d threecanvas
    let container = document.getElementById('threecanvas')
    //console.log('container size',container.clientHeight,container.clientWidth)
    this.$data.global_3d_info.cur_scene_camera = new Three.PerspectiveCamera(70, hacked_width/ hacked_height, 0.01, 10)
    this.$data.global_3d_info.cur_scene_camera.position.z = 1
    //this.$data.global_3d_info.cur_mesh = new Three.Mesh(new Three.BoxGeometry(0.4, 0.4, 0.4), new Three.MeshNormalMaterial())
    this.$data.global_3d_info.cur_scene = new Three.Scene()
    
    //this.$data.global_3d_info.cur_scene.add(this.$data.global_3d_info.cur_mesh)

    this.$data.global_3d_info.cur_renderer = new Three.WebGLRenderer({ antialias: true })
    this.$data.global_3d_info.cur_renderer.setSize(hacked_width, hacked_height)
    
    this.$data.global_3d_info.cur_renderer.setClearColor(0xffffff,1);
    
    this.$data.global_3d_info.cur_scene_control = new OrbitControls(this.$data.global_3d_info.cur_scene_camera, this.$data.global_3d_info.cur_renderer.domElement)
    this.$data.global_3d_info.cur_scene_control.rotateSpeed = 1.0
    this.$data.global_3d_info.cur_scene_control.zoomSpeed = 1.2
    this.$data.global_3d_info.cur_scene_control.panSpeed = 0.8
    this.$data.global_3d_info.cur_scene_control.enabled = true
    this.$data.global_3d_info.cur_scene_control.enableRotate = true
    this.$data.global_3d_info.cur_scene_control.enableZoom = true
    //this.$data.global_3d_info.cur_scene_control.keys = [ 65, 83, 68 ]
    
    //console.log(this.$data.global_3d_info.cur_scene)
    //console.log(this.$data.global_3d_info.cur_scene_camera)
        
    container.appendChild(this.$data.global_3d_info.cur_renderer.domElement)
    
    this.$data.download_link = document.createElement( 'a' );
    this.$data.download_link.style.display = 'none'
    document.body.appendChild( this.$data.download_link)


    this.$data.three_clock = new Three.Clock()
    this.check_model_preview()
    this.redraw_global_scene()
  
  }
};
</script>
<style lang="scss" scoped>
::-webkit-scrollbar {
  width: 0px;
  height: 1px;
}
.layer-visable{
  color: lightgray;
  opacity: 1.0;
}
.thisdropdown{
  &:hover {
    color: inherit;
    text-decoration: inherit;
  }
}
.maincanvas{
  background-color: white;
  margin-top: 100px;
  margin-left: auto;
  margin-right: auto;
}
.threecanvas{

  margin-top: 100px;
  margin-left: auto;
  margin-right: auto;
}
.layer-notvisable{
  color: gray;
  opacity: 0.6;
}
.layer-activated{
  border-color:#495057;
  color: lightgray;
  opacity: 1.0;
}
.layer-notactivated{
  border-color:lightgray;
  color: #495057;
  opacity: 1.0;
}
.el-dropdown-link {
  cursor: pointer;
  color: lightgray;
  font-size: 14px;
  height: 82px;
  width: 90%;
  border-radius: 5px;
  border-color: #dbdbdb;
  background-color:rgb(19,23,27) ;
}
.el-icon-arrow-down {
  font-size: 16px;
}
.el-dropdown-item{
    &:hover {
    opacity: 0.5;
    transition-duration:.2s;
    transition-timing-function:cubic-bezier(.02,.01,.05,1);
  }
}
</style>