/* Artemis Plasma Pro — Qwen3.5-0.8B tabanlı model, transformers.js ile tarayıcıda çalışır.
   Model dosyaları Hugging Face'ten bir kez iner, tarayıcı önbelleğinde kalır.
   WebGPU gerekir: ekran kartında q4f16 (~470 MB) ya da shader-f16 yoksa q4 (~550 MB) çalışır.
   artemis-worker.js ile aynı mesajları kullanır: load → progress/ready, generate → token/done, stop. */
import { AutoTokenizer, AutoModelForCausalLM, TextStreamer, InterruptableStoppingCriteria } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.min.js';

var MODEL = 'onnx-community/Qwen3.5-0.8B-Text-ONNX';

// Artemis kimliği ve dil kuralı: Qwen'in üzerine eklenen kısım
var SYSTEM = "Sen Artemis Plasma Pro'sun. Athene Softworks tarafından geliştirilen bir yapay zekâ asistanısın ve tamamen kullanıcının tarayıcısında, cihazında çalışırsın; yazılanlar hiçbir sunucuya gitmez.\n" +
  'Kurallar:\n' +
  '- Kullanıcı Türkçe yazarsa Türkçe, İngilizce yazarsa İngilizce yanıt ver.\n' +
  '- Kısa, açık ve doğru yanıt ver. Emin değilsen bilmediğini söyle, uydurma.\n' +
  "- Kim olduğun sorulursa: \"Ben Artemis Plasma Pro, Athene Softworks'ün yapay zekâ asistanıyım.\" de.\n" +
  "You are Artemis Plasma Pro, an AI assistant made by Athene Softworks, running on the user's device.";
// Kimlik örnekleri: küçük model sistem mesajını tek başına her zaman tutmuyor
var SHOTS = [
  { role: 'user', content: 'Sen kimsin?' },
  { role: 'assistant', content: "Ben Artemis Plasma Pro, Athene Softworks'ün geliştirdiği bir yapay zekâ asistanıyım. Tamamen sizin tarayıcınızda çalışıyorum. Size nasıl yardımcı olabilirim?" },
  { role: 'user', content: 'Who are you?' },
  { role: 'assistant', content: "I'm Artemis Plasma Pro, an AI assistant made by Athene Softworks. I run entirely in your browser. How can I help?" }
];

var tokenizer = null, model = null, loading = null, run = 0, last = Promise.resolve();
var stopper = new InterruptableStoppingCriteria();

// Yalnızca WebGPU: modelin nicelenmiş gömme katmanı (GatherBlockQuantized) WASM'da çalışmıyor
async function dtype() {
  var a = null;
  try { a = navigator.gpu && await navigator.gpu.requestAdapter(); } catch (e) {}
  if (!a) { var err = new Error('WebGPU yok'); err.code = 'nogpu'; throw err; }
  return a.features.has('shader-f16') ? 'q4f16' : 'q4';
}

function load() {
  if (loading) return loading;
  var files = {};
  function progress(p) {
    if (p.status !== 'progress' || !p.total) return;
    files[p.file] = p;
    var loaded = 0, total = 0;
    for (var f in files) { loaded += files[f].loaded; total += files[f].total; }
    postMessage({ type: 'progress', loaded: loaded, total: total });
  }
  loading = (async function () {
    var dt = await dtype();
    tokenizer = await AutoTokenizer.from_pretrained(MODEL);
    model = await AutoModelForCausalLM.from_pretrained(MODEL, { dtype: dt, device: 'webgpu', progress_callback: progress });
  })();
  loading.catch(function () { loading = null; });
  return loading;
}

async function generate(history, o) {
  var id = ++run, text = '';
  stopper.reset();
  var msgs = [{ role: 'system', content: SYSTEM }].concat(SHOTS, history.map(function (m) {
    return { role: m.role === 'bot' ? 'assistant' : 'user', content: m.text };
  }));
  var inputs = tokenizer.apply_chat_template(msgs, { add_generation_prompt: true, return_dict: true, enable_thinking: false });
  var streamer = new TextStreamer(tokenizer, {
    skip_prompt: true, skip_special_tokens: true,
    callback_function: function (t) { if (id === run && t) { text += t; postMessage({ type: 'token', text: t }); } }
  });
  await model.generate(Object.assign({}, inputs, {
    max_new_tokens: o.maxTokens, do_sample: true, temperature: o.temperature, top_k: o.topK, top_p: o.topP,
    repetition_penalty: o.repetitionPenalty, streamer: streamer, stopping_criteria: stopper
  }));
  if (id === run) postMessage({ type: 'done', text: text });
}

onmessage = function (e) {
  var m = e.data;
  function fail(err) { postMessage({ type: 'error', message: String(err.message || err), code: err.code }); }
  // Model bir kez yüklenir; başka modelden geri dönüldüğünde hemen hazır denir
  if (m.type === 'load') load().then(function () { postMessage({ type: 'ready', name: 'Artemis Plasma Pro' }); }).catch(fail);
  else if (m.type === 'stop') { run++; stopper.interrupt(); postMessage({ type: 'done', stopped: true }); }
  // Durdurulan yanıt bitmeden yenisi başlamasın: üretimler sırayla çalışır
  else if (m.type === 'generate') last = last.then(function () { return generate(m.history, m.options); }).catch(fail);
};
