const CACHE_NAME = 'panel-cache-v1';

// Liste des ressources à mettre en cache (ajoutez les fichiers externes ici)
const urlsToCache = [
  '/comit_dashboard/dashboard_commit.js',   // Votre script généré par Panel
  '/comit_dashboard/pyodide.js',                // Pyodide
  '/comit_dashboard/pyodide.wasm',              // WebAssembly de Pyodide
  'https://cdn.holoviz.org/panel/1.5.3/dist/bundled/reactiveesm/es-module-shims@^1.10.0/dist/es-module-shims.min.js',
  'https://cdn.bokeh.org/bokeh/release/bokeh-3.6.1.min.js',
  'https://cdn.bokeh.org/bokeh/release/bokeh-gl-3.6.1.min.js',
  'https://cdn.bokeh.org/bokeh/release/bokeh-widgets-3.6.1.min.js',
  'https://cdn.bokeh.org/bokeh/release/bokeh-tables-3.6.1.min.js',
  'https://cdn.holoviz.org/panel/1.5.3/dist/panel.min.js',
  
  // Ajout des nouveaux fichiers Template JS
  'https://cdn.holoviz.org/panel/1.5.3/dist/bundled/fastbasetemplate/fast_template.js',
  'https://cdn.holoviz.org/panel/1.5.3/dist/bundled/@microsoft/fast-components@2.30.6/dist/fast-components.js',
  'https://cdn.holoviz.org/panel/1.5.3/dist/bundled/fast/js/fast_design.js',

  // Ajout des fichiers CSS à mettre en cache
  'https://cdn.holoviz.org/panel/1.5.3/dist/css/loadingspinner.css',
  'https://cdn.holoviz.org/panel/1.5.3/dist/css/listpanel.css',
  'https://cdn.holoviz.org/panel/1.5.3/dist/css/button.css',
  'https://cdn.holoviz.org/panel/1.5.3/dist/css/dataframe.css',
  'https://cdn.holoviz.org/panel/1.5.3/dist/css/markdown.css',
  'https://cdn.holoviz.org/panel/1.5.3/dist/css/loading.css?v=1.5.3',
  'https://fonts.googleapis.com/css?family=Open+Sans',
  'https://cdn.holoviz.org/panel/1.5.3/dist/bundled/theme/default.css?v=1.5.3',
  'https://cdn.holoviz.org/panel/1.5.3/dist/bundled/fastbasetemplate/fast.css?v=1.5.3',
  'https://cdn.holoviz.org/panel/1.5.3/dist/bundled/fastlisttemplate/fast_list_template.css?v=1.5.3'
];

// Événement d'installation du Service Worker
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(urlsToCache);
    })
  );
});

// Événement de récupération des ressources mises en cache
// self.addEventListener('fetch', (event) => {
//   event.respondWith(
//     caches.match(event.request).then((cachedResponse) => {
//       return cachedResponse || fetch(event.request);
//     })
//   );
// });



self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then((cachedResponse) => {
      // Si la ressource est en cache, la retourner
      if (cachedResponse) {
        return cachedResponse;
      }

      // Sinon, aller chercher la ressource dans le réseau
      return fetch(event.request).then((response) => {
        // Si la réponse est valide, la mettre en cache
        if (response && response.status === 200 && response.type === 'basic') {
          const responseToCache = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, responseToCache);
          });
        }
        return response;
      });
    })
  );
});
