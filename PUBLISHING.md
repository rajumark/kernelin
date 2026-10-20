# Publishing Kernelin

Publishing uses the [vanniktech maven-publish plugin](https://vanniktech.github.io/gradle-maven-publish-plugin/).
One upload to the Maven Central Portal contains every module: `kernelin-runtime` (+ `-jvm`, `-wasm-js`),
`kernelin-webgpu` (+ `-wasm-js`), `kernelin-compiler-plugin`, `kernelin-gradle-plugin` and the plugin marker
`io.github.rajumark.kernelin.gradle.plugin`, each with sources, javadoc, POM and signatures.

Coordinates and shared POM data are in `gradle.properties` (`GROUP`, `VERSION_NAME`, `POM_*`); each module's own
`gradle.properties` sets `POM_ARTIFACT_ID`, `POM_NAME` and `POM_DESCRIPTION`.

## Secrets

In `~/.gradle/gradle.properties`, never in the repo:

```properties
mavenCentralUsername=<Central Portal user token username>
mavenCentralPassword=<Central Portal user token password>
signingInMemoryKey=<ASCII-armored GPG private key>
signingInMemoryKeyPassword=<gpg passphrase>
```

The namespace `io.github.rajumark` is verified on https://central.sonatype.com, and the GPG key (`18E1B575`) is on
keys.openpgp.org and keyserver.ubuntu.com.

## Every release

```bash
# 1. bump VERSION_NAME in gradle.properties, add a CHANGELOG entry

# 2. build and test everything
gradle clean build

# 3. dry run: publish to ~/.m2 and build a separate project that resolves Kernelin only from there
gradle publishToMavenLocal

# 4. upload, check the deployment at https://central.sonatype.com/publishing, then publish it
gradle publishToMavenCentral

# 5. once live (10-30 min), build a separate project against Maven Central only, then tag
git tag v0.4.0 && git push --tags
```

Maven Central versions are permanent: a published version can never be changed or deleted.
