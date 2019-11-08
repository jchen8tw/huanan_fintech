module.exports = {
    entry: [
        './src/index.jsx'
    ],
    
    module: {
        rules: [
            {
                test: /\.(js|jsx)$/,
                exclude: /node_modules/,
                loader: 'babel-loader',
                options: {
                  presets: [
                    '@babel/preset-env',
                    {
                      plugins: [
                        '@babel/plugin-proposal-class-properties'
                      ]
                    }
                  ]
                },
            },
            {
                test: /\.css$/,
                use: [
                  'style-loader',
                  {
                    loader: 'css-loader',
                    options: {
                      importLoaders: 1,
                      modules: true
                    }
                  }
                ],
                include: /\.module\.css$/
              },
              {
                test: /\.css$/,
                use: [
                  'style-loader',
                  'css-loader'
                ],
                exclude: /\.module\.css$/
              }
        ]
    },
    output: {
        path: __dirname + '/static',
        filename: 'bundle.js'
    },
};
