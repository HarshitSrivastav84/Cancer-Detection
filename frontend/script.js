// Same origin when served by Flask. Use "http://localhost:5000/predict" if opening index.html directly.
const API_URL="/predict";
const $=id=>document.getElementById(id);
const lens=$("lens"),file=$("file"),go=$("go"),clear=$("clear");
let sel=null,thumb="",last=null;
const HEX={"Low signal":[47,125,87],"Moderate signal":[192,128,26],"High signal":[184,54,90]};

function toDataURL(url,max=480){return new Promise(r=>{const i=new Image();i.onload=()=>{const k=Math.min(1,max/Math.max(i.width,i.height));const c=document.createElement("canvas");c.width=i.width*k;c.height=i.height*k;c.getContext("2d").drawImage(i,0,0,c.width,c.height);r({url:c.toDataURL("image/jpeg",.85),w:c.width,h:c.height})};i.onerror=()=>r(null);i.src=url})}
function band(p){return p<30?["Low signal","var(--low)"]:p<60?["Moderate signal","var(--mid)"]:["High signal","var(--high)"]}
function showErr(m){const e=$("err");e.textContent=m;e.style.display="block"}
function reset(){sel=null;file.value="";lens.style.backgroundImage="";lens.classList.remove("has");$("fname").textContent="";go.disabled=clear.disabled=true;$("out").style.display="none";$("empty").style.display="block";$("err").style.display="none"}

function pick(f){
  if(!f||!f.type.startsWith("image/")){showErr("Choose an image file (JPG, PNG or TIFF).");return}
  sel=f;thumb=URL.createObjectURL(f);
  lens.style.backgroundImage=`url(${thumb})`;lens.classList.add("has");
  $("fname").textContent=f.name;go.disabled=clear.disabled=false;
  $("err").style.display="none";$("out").style.display="none";$("empty").style.display="block";
}
lens.addEventListener("click",()=>file.click());
lens.addEventListener("keydown",e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();file.click()}});
file.addEventListener("change",()=>pick(file.files[0]));
["dragenter","dragover"].forEach(v=>lens.addEventListener(v,e=>{e.preventDefault();lens.classList.add("over")}));
["dragleave","drop"].forEach(v=>lens.addEventListener(v,e=>{e.preventDefault();lens.classList.remove("over")}));
lens.addEventListener("drop",e=>pick(e.dataTransfer.files[0]));
clear.addEventListener("click",reset);

go.addEventListener("click",async()=>{
  if(!sel)return;
  go.disabled=true;go.textContent="Analyzing…";lens.classList.add("busy");$("err").style.display="none";
  try{
    const fd=new FormData();fd.append("image",sel);
    const [res]=await Promise.all([fetch(API_URL,{method:"POST",body:fd}),new Promise(r=>setTimeout(r,900))]);
    const d=await res.json();
    if(!res.ok)throw new Error(d.error||"The server returned an error.");
    const p=d.cancer_probability_percent,[label,color]=band(p),cancer=d.prediction==="cancer";
    last={name:sel.name,time:new Date(),p,label,cancer,conf:d.confidence_percent,msg:d.message,disc:d.disclaimer,img:await toDataURL(thumb)};
    $("empty").style.display="none";$("out").style.display="block";
    const b=$("band");b.textContent=label;b.style.background=color;
    $("vTitle").textContent=cancer?"Tumor features detected":"No strong tumor features";
    $("vMsg").textContent=d.message;
    $("sPred").textContent=cancer?"Tumor":"Non-tumor";
    $("sConf").textContent=d.confidence_percent.toFixed(1)+"%";
    $("disc").textContent=d.disclaimer;
    $("pct").textContent=p.toFixed(1)+"%";
    const arc=$("arc");arc.style.stroke=color;arc.setAttribute("stroke-dasharray","0 100");
    requestAnimationFrame(()=>requestAnimationFrame(()=>arc.setAttribute("stroke-dasharray",p+" 100")));
    $("hist").hidden=false;
    const li=document.createElement("li");
    li.innerHTML=`<img alt=""><span class="n"></span><span class="dot"></span><span class="p"></span>`;
    li.querySelector("img").src=thumb;li.querySelector(".n").textContent=sel.name;
    li.querySelector(".dot").style.background=color;li.querySelector(".p").textContent=p.toFixed(1)+"%";
    $("list").prepend(li);
  }catch(err){
    showErr(err instanceof TypeError?"Could not reach the server. Check that app.py is running.":err.message);
  }finally{
    lens.classList.remove("busy");go.disabled=false;go.textContent="Analyze image";
  }
});

$("dl").addEventListener("click",()=>{
  if(!last)return;
  if(!window.jspdf){showErr("PDF library could not load. Check your internet connection and reload.");return}
  const {jsPDF}=window.jspdf,doc=new jsPDF({unit:"mm",format:"a4"}),W=210,M=18;
  const hema=[36,27,74],eosin=[233,163,188],ink=[30,24,56],muted=[108,100,133],col=HEX[last.label];
  doc.setFillColor(...hema);doc.rect(0,0,W,30,"F");
  doc.setFillColor(...eosin);doc.circle(M+5,15,5,"F");
  doc.setTextColor(255,255,255);doc.setFont("times","bold");doc.setFontSize(20);doc.text("HistoScreen",M+14,17);
  doc.setFont("helvetica","normal");doc.setFontSize(9);doc.setTextColor(200,192,230);
  doc.text("Gastric tissue screening report",W-M,13,{align:"right"});
  doc.text(last.time.toLocaleString(),W-M,19,{align:"right"});
  let y=42;
  if(last.img){
    const box=70,k=Math.min(box/last.img.w,box/last.img.h),w=last.img.w*k,h=last.img.h*k;
    doc.setDrawColor(...hema);doc.setLineWidth(.6);doc.rect(M-.5,y-.5,w+1,h+1);
    doc.addImage(last.img.url,"JPEG",M,y,w,h);
    doc.setFontSize(8);doc.setTextColor(...muted);doc.text(doc.splitTextToSize(last.name,box),M,y+h+5);
  }
  const X=M+82;
  doc.setFont("helvetica","bold");doc.setFontSize(9);doc.setTextColor(...muted);doc.text("TUMOR PROBABILITY",X,y+4);
  doc.setFont("times","bold");doc.setFontSize(38);doc.setTextColor(...col);doc.text(last.p.toFixed(1)+"%",X,y+20);
  doc.setFillColor(...col);doc.roundedRect(X,y+25,38,7,3.5,3.5,"F");
  doc.setFont("helvetica","bold");doc.setFontSize(9);doc.setTextColor(255,255,255);doc.text(last.label,X+19,y+29.8,{align:"center"});
  doc.setFillColor(235,228,240);doc.roundedRect(X,y+38,W-M-X,4,2,2,"F");
  doc.setFillColor(...col);doc.roundedRect(X,y+38,Math.max(1,(W-M-X)*last.p/100),4,2,2,"F");
  const rows=[["Model prediction",last.cancer?"Tumor":"Non-tumor"],["Model confidence",last.conf.toFixed(1)+"%"],["Model","EfficientNet-B0, binary classifier"]];
  let ry=y+53;
  rows.forEach(r=>{doc.setFont("helvetica","normal");doc.setFontSize(9);doc.setTextColor(...muted);doc.text(r[0],X,ry);
    doc.setFont("helvetica","bold");doc.setTextColor(...ink);doc.text(r[1],X+36,ry);ry+=7});
  y=Math.max(y+85,ry+10);
  doc.setFont("times","bold");doc.setFontSize(14);doc.setTextColor(...ink);
  doc.text(last.cancer?"Tumor features detected":"No strong tumor features",M,y);
  doc.setFont("helvetica","normal");doc.setFontSize(10);doc.setTextColor(...ink);
  y+=7;const m=doc.splitTextToSize(last.msg,W-2*M);doc.text(m,M,y);y+=m.length*5+8;
  const dtxt=doc.splitTextToSize(last.disc+" This report is generated by a research prototype trained on a single public dataset of 224x224 patches. It may flag healthy tissue as tumor and may miss real tumors.",W-2*M-10);
  const bh=dtxt.length*4.6+16;
  doc.setFillColor(250,245,248);doc.setDrawColor(...eosin);doc.roundedRect(M,y,W-2*M,bh,2,2,"FD");
  doc.setFont("helvetica","bold");doc.setFontSize(9);doc.setTextColor(...hema);doc.text("Important",M+5,y+7);
  doc.setFont("helvetica","normal");doc.setFontSize(9);doc.setTextColor(...ink);doc.text(dtxt,M+5,y+13);
  doc.setFontSize(8);doc.setTextColor(...muted);
  doc.text("Not a medical diagnosis. Review by a qualified pathologist is required.",W/2,288,{align:"center"});
  const stamp=last.time.toISOString().slice(0,16).replace(/[:T]/g,"-");
  doc.save(`histoscreen-report-${stamp}.pdf`);
});