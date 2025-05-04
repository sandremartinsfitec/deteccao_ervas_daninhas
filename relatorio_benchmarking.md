# Relatório de Benchmarking: Aplicação Streamlit para Classificação Agrícola

## 1. Introdução

Este relatório apresenta uma análise comparativa (benchmarking) da aplicação web desenvolvida em Streamlit para classificação de imagens agrícolas, focada na identificação de ervas daninhas em lavouras de soja, frente a soluções comerciais existentes no mercado. O objetivo é avaliar o posicionamento da aplicação desenvolvida, identificando seus pontos fortes, limitações e potenciais diferenciais em relação a ferramentas estabelecidas.

A aplicação desenvolvida utiliza um modelo ResNet50 pré-treinado para classificar imagens em quatro categorias (Soja, Folha Larga, Grama, Solo) e implementa três métodos de localização aproximada de ervas daninhas (Heatmap via CAM, Janela Deslizante e Bounding Boxes via NMS sobre as detecções da janela deslizante), oferecendo uma interface interativa em português para upload e análise de imagens.

## 2. Metodologia

A análise comparativa foi realizada através de pesquisa web, identificando softwares e plataformas que oferecem funcionalidades similares, especificamente a detecção e mapeamento de ervas daninhas utilizando imagens aéreas (principalmente de drones). Foram selecionadas três soluções representativas do mercado para análise mais detalhada:

1.  **FlyPix AI:** Plataforma geoespacial com IA para diversas indústrias, incluindo agricultura.
2.  **Pix4Dfields:** Software especializado em mapeamento agrícola com drones.
3.  **BemAgro:** Plataforma SaaS focada em agricultura inteligente com IA.

A comparação considerou aspectos como: funcionalidades principais, tecnologia utilizada, foco (classificação vs. detecção vs. mapeamento completo), facilidade de uso, customização, modelo de negócio (software, SaaS, API) e público-alvo.

## 3. Visão Geral da Aplicação Desenvolvida

*   **Plataforma:** Aplicação Web (Streamlit)
*   **Tecnologia Principal:** Python, Streamlit, PyTorch (ResNet50), OpenCV, PIL
*   **Funcionalidades:**
    *   Upload de imagens (JPG, PNG, TIFF).
    *   Classificação da imagem inteira (Soja, Folha Larga, Grama, Solo).
    *   Visualização de probabilidades por classe.
    *   Localização aproximada de ervas daninhas (Folha Larga, Grama) via:
        *   Heatmap (Class Activation Mapping - CAM).
        *   Janela Deslizante (com visualização de patches detectados).
        *   Bounding Boxes (agregando detecções da janela deslizante com Non-Maximum Suppression - NMS).
    *   Interface interativa em português com parâmetros ajustáveis (thresholds).
*   **Modelo:** ResNet50 (pré-treinado, fornecido pelo usuário).
*   **Público-Alvo:** Pesquisadores, desenvolvedores, agrônomos que necessitam de uma ferramenta visual e interativa para análise pontual de imagens, com flexibilidade para usar modelos próprios.
*   **Modelo de Negócio:** Código aberto (potencial), execução local ou em servidor próprio.

## 4. Análise das Soluções Comerciais

### 4.1. FlyPix AI

*   **Descrição:** Plataforma de IA geoespacial que utiliza imagens aéreas (drones, satélites) para detecção de objetos em diversas indústrias, incluindo agricultura (monitoramento de culturas, contagem de plantas, detecção de ervas daninhas) e pecuária (contagem e monitoramento de gado).
*   **Funcionalidades Relevantes:** Detecção e mapeamento de crescimento de ervas daninhas, análise de saúde da cultura, detecção de anomalias.
*   **Tecnologia:** Plataforma web (provavelmente SaaS), IA (Machine Learning/Deep Learning) para processamento de imagem.
*   **Diferenciais:** Foco amplo em detecção de objetos diversos, integração com diferentes fontes de imagem, análise geoespacial avançada.
*   **Limitações (Inferidas):** Menos focado exclusivamente em agricultura que outras soluções, pode ser uma plataforma mais complexa e cara, modelo de IA provavelmente proprietário ("caixa preta").
*   **URL:** https://flypix.ai/pt/object-detection-software-for-agriculture-and-farming/

### 4.2. Pix4Dfields

*   **Descrição:** Software de mapeamento agrícola avançado projetado para análise aérea de culturas com drones. Focado em transformar imagens em insights acionáveis para agricultura de precisão.
*   **Funcionalidades Relevantes:** Criação de ortomosaicos e mapas de índices (NDVI, etc.), mapas de saúde da cultura, geração de mapas de prescrição para pulverização localizada, ferramenta "Magic Tool" (IA) para detecção e seleção de ninhos de ervas daninhas, planejamento de voos de pulverização.
*   **Tecnologia:** Software desktop e/ou cloud, processamento fotogramétrico, IA para funcionalidades específicas (Magic Tool).
*   **Diferenciais:** Especializado em agricultura, fluxo de trabalho completo desde o voo até o mapa de prescrição, integração com maquinário agrícola, processamento rápido otimizado para campo.
*   **Limitações (Inferidas):** Software pago (licença), pode exigir hardware robusto, foco maior em mapeamento e prescrição do que em classificação detalhada por imagem individual.
*   **URL:** https://www.pix4d.com/pt/industria/agricultura, https://www.pix4d.com/pt/blog/pix4dfields-magic-tool

### 4.3. BemAgro

*   **Descrição:** Plataforma SaaS (Software as a Service) que utiliza IA para fornecer soluções para agricultura inteligente, cobrindo processamento, planejamento, monitoramento e controle.
*   **Funcionalidades Relevantes:** Módulo "Monitoring" com IA para detectar plantas daninhas e gerar zonas de manejo para aplicação em taxa variável.
*   **Tecnologia:** Plataforma web (SaaS), IA.
*   **Diferenciais:** Modelo SaaS (acessível via navegador, atualizações contínuas), foco em gerar zonas de manejo práticas para aplicação localizada, integração de diferentes etapas do ciclo agrícola.
*   **Limitações (Inferidas):** Dependência de assinatura, modelo de IA proprietário, pode ter menos flexibilidade para customização pelo usuário final.
*   **URL:** https://www.bemagro.com/

## 5. Análise Comparativa

| Característica          | Aplicação Desenvolvida                     | FlyPix AI                     | Pix4Dfields                   | BemAgro                      |
| :---------------------- | :----------------------------------------- | :---------------------------- | :---------------------------- | :--------------------------- |
| **Plataforma**          | Web App (Streamlit)                        | Plataforma Web (SaaS prov.)   | Software Desktop/Cloud        | Plataforma Web (SaaS)        |
| **Foco Principal**      | Classificação e Localização Visual       | Detecção de Objetos (Amplo) | Mapeamento e Prescrição     | Zonas de Manejo (SaaS)       |
| **Detecção Ervas**    | Heatmap, Janela Deslizante, BBox (NMS)   | Sim (IA)                      | Sim (Magic Tool - IA)         | Sim (IA)                     |
| **Modelo IA**           | ResNet50 (Customizável)                    | Proprietário                  | Proprietário (Magic Tool)     | Proprietário                 |
| **Interface**           | Interativa, Simples (Streamlit)            | Profissional, Complexa prov.  | Profissional, Especializada | Profissional, Web (SaaS)     |
| **Customização**        | Alta (Código aberto potencial)             | Baixa                         | Média (Parâmetros)          | Baixa                        |
| **Custo**               | Baixo (Infraestrutura)                     | Assinatura/Licença (Alto prov.) | Licença (Software)          | Assinatura (SaaS)            |
| **Facilidade de Uso**   | Média (Requer setup inicial)             | Média/Alta (Plataforma)       | Média (Software específico) | Alta (SaaS)                  |
| **Output Principal**    | Imagem classificada + Overlays visuais     | Mapas, Detecções              | Mapas, Zonas, Prescrições   | Zonas de Manejo              |

## 6. Pontos Fortes da Aplicação Desenvolvida

*   **Leveza e Acessibilidade:** Sendo uma aplicação Streamlit, é relativamente leve e pode ser executada localmente ou em servidores com configuração mínima, sendo acessível via navegador.
*   **Flexibilidade do Modelo:** Permite a utilização de modelos PyTorch customizados (como o ResNet50 fornecido), oferecendo flexibilidade para pesquisa e experimentação.
*   **Múltiplos Métodos de Visualização:** Oferece três abordagens distintas (Heatmap, Janela Deslizante, Bounding Box) para a localização visual de ervas daninhas, permitindo diferentes perspectivas de análise.
*   **Código Aberto (Potencial):** A natureza do desenvolvimento permite que o código seja aberto, facilitando a customização, auditoria e contribuição da comunidade.
*   **Interatividade:** A interface Streamlit permite ajustes de parâmetros (thresholds) em tempo real, facilitando a exploração dos resultados.
*   **Custo:** O custo principal está associado à infraestrutura para execução, sendo potencialmente muito menor que licenças de software ou assinaturas SaaS.

## 7. Limitações da Aplicação Desenvolvida

*   **Escopo Limitado:** Focada na análise de imagens individuais, não oferece funcionalidades de mapeamento, processamento em lote de grandes áreas ou geração de mapas de prescrição como as soluções comerciais.
*   **Setup e Manutenção:** Requer conhecimento técnico para configurar o ambiente Python, dependências e executar a aplicação. A atualização e manutenção dependem do desenvolvedor.
*   **Robustez da Detecção:** O método de Bounding Box é uma aproximação baseada em NMS sobre a janela deslizante, não utilizando um modelo de detecção de objetos dedicado, o que pode ser menos preciso que soluções comerciais com IA específica para detecção.
*   **Interface de Usuário:** Embora funcional, a interface Streamlit pode ser menos polida e ter menos recursos de gerenciamento de dados que plataformas SaaS ou softwares desktop dedicados.
*   **Escalabilidade:** A arquitetura Streamlit pode ter limitações de escalabilidade para processar um volume massivo de imagens ou usuários simultâneos em comparação com plataformas cloud nativas.
*   **Modelo "Caixa Branca" vs. "Caixa Preta":** Enquanto a flexibilidade do modelo é um ponto forte, a necessidade de fornecer um modelo treinado pode ser uma barreira para usuários não técnicos, que podem preferir a simplicidade de uma solução "caixa preta" comercial.

## 8. Conclusão e Recomendações

A aplicação Streamlit desenvolvida se posiciona como uma ferramenta valiosa para **análise visual e interativa** de imagens agrícolas individuais, especialmente em contextos de **pesquisa, desenvolvimento e validação de modelos**. Seus pontos fortes residem na flexibilidade, baixo custo potencial e na oferta de múltiplos métodos de visualização para localização de ervas daninhas.

No entanto, não compete diretamente com as funcionalidades de **mapeamento em larga escala, geração de prescrição e gestão integrada** oferecidas por plataformas comerciais como Pix4Dfields ou BemAgro. Estas são soluções mais completas para agricultura de precisão operacional.

**Recomendações para Evolução:**

1.  **Processamento em Lote:** Implementar a capacidade de processar múltiplas imagens em segundo plano.
2.  **Modelo de Detecção:** Integrar um modelo de detecção de objetos (como YOLO, Faster R-CNN) treinado especificamente para ervas daninhas, substituindo ou complementando a abordagem atual de BBox via NMS/SW.
3.  **Georreferenciamento:** Adicionar suporte para imagens georreferenciadas e exibir resultados em um mapa interativo (e.g., com Folium ou Pydeck).
4.  **Empacotamento:** Facilitar a distribuição e instalação, talvez via Docker.
5.  **Comparação de Métodos:** Permitir a visualização lado a lado dos resultados dos três métodos de localização para facilitar a comparação.

## 9. Referências

*   FlyPix AI: https://flypix.ai/pt/object-detection-software-for-agriculture-and-farming/
*   Pix4Dfields:
    *   https://www.pix4d.com/pt/industria/agricultura
    *   https://www.pix4d.com/pt/blog/pix4dfields-magic-tool
*   BemAgro: https://www.bemagro.com/
*   Outras fontes consultadas na pesquisa inicial:
    *   https://mappa.ag/blog/drones-controle-de-plantas-daninhas/
    *   https://www.alavoura.com.br/colunas/startups/tecnologia-identifica-pragas-e-ervas-daninhas-com-imagens-de-drone-e-satelite/
    *   https://geoagri.com.br/public/blog/16/combate-as-plantas-daninhas-utilizando-o-processamento-de-imagens

