import { cleanEnv, str, port, makeValidator } from "envalid";
import dotenv from "dotenv";

// Env loading:
dotenv.config({ path: `.env.${process.env.NODE_ENV}` });

// Custom validators:
const whitelistValidator = makeValidator(whitelistValidatorCallback);

// Clean ENV:
export default cleanEnv(process.env, {
    PORT: port(),
    WHITELIST: whitelistValidator(),
    NODE_ENV: str({ choices: ["dev", "prod"] })
});

export function whitelistValidatorCallback(x: unknown) {
    if (!x) throw new Error("Missing value");
    if (typeof x !== "string") throw new Error("Value must be a string");
    if (x.slice(-1) === ";" || x[0] === ";")
        throw new Error("String cannot start or end with ;");
    const origins: string[] = [];
    try {
        x.split(";").forEach((url) => {
            origins.push(new URL(url).origin);
        });
    } catch (error) {
        throw new Error(
            "String contains invalid URL, Required format: origin;origin;...;origin"
        );
    }
    return origins;
}
