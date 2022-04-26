module.exports = {
    lintOnSave: undefined,
    publicPath: './',
    outputDir: undefined,
    assetsDir: undefined,
    runtimeCompiler: undefined,
    productionSourceMap: undefined,
    parallel: false,
    css: undefined
}

// const defineConfig = require('@vue/cli-service')
// // 默认配置
// module.exports = defineConfig({
//     transpileDependencies: true,
//     // 跨域支持
//     devServer: {
//         proxy: {
//             '/api': {
//                 target: `http://127.0.0.1:8000/api`,
//                 changeOrigin: true,
//                 pathRewrite: {
//                     '^/api': ''
//                 }
//             }
//         }
//     }
// })