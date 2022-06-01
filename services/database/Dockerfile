FROM postgres
EXPOSE 5432
ENV POSTGRES_USERNAME art_user
ENV POSTGRES_PASSWORD art_class
ENV POSTGRES_DB art_db
COPY art_db.sql /docker-entrypoint-initdb.d/