#/bin/bash
source .env.$ENVIRONMENT

# Process all configuration templates
shopt -s globstar 
for template in model_repository/**/*.template; do
    output="${template%.template}"
    envsubst < "$template" > "$output"
    echo "Generated $output from $template"
done