// HireFlow Copilot on Azure Container Apps, with logs and traces in Application Insights.
//
//   az group create -n rg-hireflow-copilot -l westeurope
//   az deployment group create -g rg-hireflow-copilot -f infra/azure/main.bicep \
//     -p image=ghcr.io/autumntrace/hireflow-copilot:latest anthropicApiKey=<key>
//
// Leave anthropicApiKey empty to deploy in demo mode (no model calls, no spend).

@description('Azure region for every resource.')
param location string = resourceGroup().location

@description('Prefix for resource names.')
param name string = 'hireflow-copilot'

@description('Container image to run.')
param image string = 'ghcr.io/autumntrace/hireflow-copilot:latest'

@secure()
@description('Claude API key. Empty means demo mode.')
param anthropicApiKey string = ''

@secure()
@description('Optional shared secret callers must send as X-Copilot-Key.')
param accessKey string = ''

@description('Global spend ceiling per UTC day, in USD.')
param dailyBudgetUsd string = '1.00'

resource logs 'Microsoft.OperationalInsights/workspaces@2023-09-01' = {
  name: '${name}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource insights 'Microsoft.Insights/components@2020-02-02' = {
  name: '${name}-insights'
  location: location
  kind: 'web'
  properties: {
    Application_Type: 'web'
    WorkspaceResourceId: logs.id
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: '${name}-env'
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logs.properties.customerId
        sharedKey: logs.listKeys().primarySharedKey
      }
    }
  }
}

var secrets = concat(
  [{ name: 'appinsights-connection', value: insights.properties.ConnectionString }],
  empty(anthropicApiKey) ? [] : [{ name: 'anthropic-api-key', value: anthropicApiKey }],
  empty(accessKey) ? [] : [{ name: 'access-key', value: accessKey }]
)

var env = concat(
  [
    { name: 'COPILOT_OTEL_EXPORTER', value: 'azure' }
    { name: 'APPLICATIONINSIGHTS_CONNECTION_STRING', secretRef: 'appinsights-connection' }
    { name: 'COPILOT_DAILY_BUDGET_USD', value: dailyBudgetUsd }
    // Container Apps' ingress sets X-Forwarded-For; trust it so rate limits see real client IPs.
    { name: 'COPILOT_TRUST_PROXY', value: 'true' }
  ],
  empty(anthropicApiKey) ? [] : [{ name: 'ANTHROPIC_API_KEY', secretRef: 'anthropic-api-key' }],
  empty(accessKey) ? [] : [{ name: 'COPILOT_ACCESS_KEY', secretRef: 'access-key' }]
)

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: name
  location: location
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
      }
      secrets: secrets
    }
    template: {
      containers: [
        {
          name: 'copilot'
          image: image
          env: env
          resources: {
            cpu: json('1.0')
            memory: '2Gi'
          }
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 20
            }
          ]
        }
      ]
      scale: {
        // Rate limits and the spend ledger live in process memory. One replica keeps them exact,
        // and never scaling to zero means a restart can't quietly reset today's spend. Move that
        // state to Redis (see guard.py) before scaling out. Also set a monthly spend limit on the
        // key in the Anthropic Console as the backstop.
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

output url string = 'https://${app.properties.configuration.ingress.fqdn}'
